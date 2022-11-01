import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    print("Model:")
    print(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            # need to install warpctc. see our guideline.
            from warpctc_pytorch import CTCLoss 
            criterion = CTCLoss()
        else:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter

    while(True):
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                cost = criterion(preds, text, preds_size, length) / batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)

        else:
            preds = model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)

        # validation part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            elapsed_time = time.time() - start_time
            # for log
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                model.train()

                # training loss and validation loss
                loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 1e+5 iter.
        if (iteration + 1) % 1e+5 == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.exp_name}/iter_{iteration+1}.pth')

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='/',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='1',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='골 목미용실한성부동산홍라운지모단걸응접씨앗양식하정소의원화가구백점마훠궈오로클래츠반상저온숙수제전문치과유황리참숯불우림자작나무금강포장중기조헤어샵까끄아영빈케익스활선약국밀돼밥세신피노키경이진철거주필립터디카페빙그레트바머연출윤네멸쿡즈데예건축인월천공물꽃니쿨방설집옥최고등급드곤보울야토탈녀복렌창업심콜렉션테내몸에손탁텔린취는음악학른프임호희망개사찬명통증교크승족핸맨닥옷만혜빌민패꼬김당티타감각더벽글벌여행쌤님돈퍼관탐습코푸도차현계일채폰센애플서쉼생술쿠르따뜻송커힐대쉐놀갈비객즐다킨믹루큰워뷰직력곰맥할맛있회올안환베남쌀랑콩뜰달북먹발틈새극재록란쭈꾸찌판매능샘블속독법분롱짐본엄냉닭합염군탑광열엔류튜살헨느빠풍시요게랜엘틸종샷와박규형외딩흉충두파벨또름배청친효돌쌈길틱을앤쓰담잔위쏠힘쎄젤싸빵료은퀼컷꼴투콤향핀초십삼짇날씽련택해석탄팽떼쎼닛딸링젠캔버봄샤뮤엠메칼탕슉번째샐침태후랄늘러브휴냥표볶넬쁘곱쫄댄릇빗쥬얼육체혼좋으뜸누떡낭겹닉품든옛팬밤삭잉면즉끓총별빛춘셀큐흥막럽곡준턴샹릴귀델팔득슈끼평싶컬캐롬눔욕빨꺼책핑줄밴순햇앙촌너허웰핏낙쇄밍결헬훈탭굽홈권닝뉴템꿈켈람층릉알휘항간곽근풀편콘텐역핌룩팥봉룸붐찜퓨덕컴완펜댕숭톰엌럴깔끔샬협쿄함입뎅쩐둥솔횟왕율쌍멋칠앵때릭견특깃짬뽕갤난례멀획컨팅량멍텅릿쇼룹샌닮언왁싱눈썹꾼추털뜨락쉬숍쵸팡듬짝맵랙잠갑께뽀흑색흠둑쇠뇌혈암쏘펠빼찻멘들히롯년끌켓꿀뻘잇옆착븐룡쎈깨슬홀믿몽쏭핫펌볼륨캡굿짱끝뽈쑝뚝딱짜엣병숲널몰써덴던뭄죽궁낵긴튬웃뭉텀겐펍댁쭌칭멜벤처땅랩붙검킴솥엉범윈쟁낚샨똘킹격렁된롤적뇽썰잡를퀸몬넷존넘액윌뇨셉농펫팰며튀압뢰웨덤쉽받싼런팩욱밸밭져썸맹뱃쁨윗옴므령돔맘틀럭륜말셋뷔논읽닐험융려떤척닷밝텍턱룻짚넌첨씬절풋묘곳납왓옵픽퀘캠질렐톡젼헛넥흰뱅쁠띠깐뎃탠밌칸빅톤팝률옻벧확듀덮퉁섬딘즌튼넝꼼캣론럼쉴측젝쿱흡맑잼팀엑렛웹죠딕곁잎좌컵퀵떳쾌높딧벡놔쫌얄렙윙쳐변뎀즘랭없렬넣흔톨녹맞춤굴및움뚱잭솟밧픈찰슨롭삽녕붕슐쿤쟈맴뽁켐솜팜긋숨젬억졸셨첫뒤될셜찹빔됩챌꽁먼뚜펙킥뒷깜빡헌폐캬뽑삘툴겨땡떙럿겁렘냄곧셔흐균힌즙첸많념쩡칡징딴햄뼈웅춧퇴벚킬뻐썬츄촬옳껍벗앰묵찾뵙겠랫폼샾램렵숴봐갓넓떠앞뚫닙켄되싯왔씻잘섯략않것꼰혹긍답짧폴켑랐칙핵폭혁왜묻냐엇떻괴했겅짓삶겔덩훙촛듯쓴옮멧큘싫씩괜찮죄혐걱낳았뭐둠퀴쓸끈었렇겼꿔멈좁뭔딜잃봤앓팍햬헥귝얀듦같씀뿌엽훔걷깊휼엿냈쁜놓밖큼낫닌맙봇닫둔뀐캇듣붓랍켜숫잊혀틴튤못벼떨긷뛰슴찢맺줍쪽붉첩늙귓괘갔헝섹쥐낮챔캉섭쯤좀줘졌딥캘옙걀둘덟씹헸웠볍였픔셰잖낸핍냠넙뻔콕엮얻솝챙랬펀뮈셸쉿뻬흘꽂칵몇툰깥꼭낯탉놈빚늑줌첼삿뮌찐햅펭귄빽옹댓맷퀀잌끗꽈넛숩숀슥쉘룽싹륭훼왼흙똥갱겟콧뮬헐룰믈푼샛벳쑈퐁읍쿼벅앨뫼뗌롶껑웬삐톱값딤쁄훌젓펄줏쏙짙텝볕궐렴펑뉘탱륙꿉꾹얌떴쿰늄밑숑킷돗팻귤뀨욜뤼셴똑꺽웜꿍쿵돝깡쥴밋겸뱍띵쪼쌩갯젊텃헉랏굼랴똔깻긱팟휠덱찍멤깍븟멕쑥뺑쟝궃촉샴쏨녘갸윰픙뱀떄벙늪뿔뿍껏숖홉듸숟븍셈썅묭밈쯔맟캄낌탤쓱릎섞찡굯쏜솣쫀꼽굉돋묶씰힙쥰삥챠땐껴햐탬쭉횡촨뽄옾땀곶굄멩껌챈꾀켠룬잣텬웁깅윔낼쑤툼륵둣얏녜돛풉챨쭝엥깎욘뙈갭녁쪄훗웍덜뺴듭튠쮸얹쉔뻥괸씸쏀왈쨈섀옐갖뢍괄툇짠댑헹캅뀀껄휀똣넋빳틔댐띄헵꽐썽쳇눌쎌떢렷롸넉힝뿐얜쑨챤뵈볏묾쩌깽쟌뿜랠칫죤뺀캥댥퀄렝뭇챕칩뺭뽐맬샥냇퇘왠켘쌉횃댜섶몀큽펴얇슌슁끊엎슘졍돐숄넵톳팸젯빻붎꼐셩읜햔꿩댈딪늬빤잽깝볽됨쏴뛴컹땍틉끽꺠앉뤠릅쯴늠웸짤뻑뼝쫑쩍갬뗑앱뜌뜜쩜쌔섧잗뗴넨쨍껀릐눕뷜낄푹슟뭘꼈낀뚤뜬뜯삑냅뀝횰쩨쎔쌓놋샆췬힉싀헙앳껨덧윷겜놉싣뎁캌윅팁펏났옌펼핯훨썩뫔궂잿촘췄뚠땁얘튈깁봅짭벵킵넴빱엡쇳쉰딍띤삔갠콥쿳싁럷툽틍젖', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--tp', default='vert', help='the type of model')

    opt = parser.parse_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.tp}-{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    train(opt)
