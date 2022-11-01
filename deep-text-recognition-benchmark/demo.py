import string
import argparse
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
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
    # model = torch.nn.DataParallel(model).to(device)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)
    save_data = []
    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)


            log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            # print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                # print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                img_name = './'+img_name
                save_data.append([img_name, pred])
            df = pd.DataFrame(save_data, columns=['img_path', 'text'])
            df.to_csv(f'./{opt.tp}_result.csv', index=None, encoding='utf-8-sig')
            log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=64, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=200, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='골 목미용실한성부동산홍라운지모단걸응접씨앗양식하정소의원화가구백점마훠궈오로클래츠반상저온숙수제전문치과유황리참숯불우림자작나무금강포장중기조헤어샵까끄아영빈케익스활선약국밀돼밥세신피노키경이진철거주필립터디카페빙그레트바머연출윤네멸쿡즈데예건축인월천공물꽃니쿨방설집옥최고등급드곤보울야토탈녀복렌창업심콜렉션테내몸에손탁텔린취는음악학른프임호희망개사찬명통증교크승족핸맨닥옷만혜빌민패꼬김당티타감각더벽글벌여행쌤님돈퍼관탐습코푸도차현계일채폰센애플서쉼생술쿠르따뜻송커힐대쉐놀갈비객즐다킨믹루큰워뷰직력곰맥할맛있회올안환베남쌀랑콩뜰달북먹발틈새극재록란쭈꾸찌판매능샘블속독법분롱짐본엄냉닭합염군탑광열엔류튜살헨느빠풍시요게랜엘틸종샷와박규형외딩흉충두파벨또름배청친효돌쌈길틱을앤쓰담잔위쏠힘쎄젤싸빵료은퀼컷꼴투콤향핀초십삼짇날씽련택해석탄팽떼쎼닛딸링젠캔버봄샤뮤엠메칼탕슉번째샐침태후랄늘러브휴냥표볶넬쁘곱쫄댄릇빗쥬얼육체혼좋으뜸누떡낭겹닉품든옛팬밤삭잉면즉끓총별빛춘셀큐흥막럽곡준턴샹릴귀델팔득슈끼평싶컬캐롬눔욕빨꺼책핑줄밴순햇앙촌너허웰핏낙쇄밍결헬훈탭굽홈권닝뉴템꿈켈람층릉알휘항간곽근풀편콘텐역핌룩팥봉룸붐찜퓨덕컴완펜댕숭톰엌럴깔끔샬협쿄함입뎅쩐둥솔횟왕율쌍멋칠앵때릭견특깃짬뽕갤난례멀획컨팅량멍텅릿쇼룹샌닮언왁싱눈썹꾼추털뜨락쉬숍쵸팡듬짝맵랙잠갑께뽀흑색흠둑쇠뇌혈암쏘펠빼찻멘들히롯년끌켓꿀뻘잇옆착븐룡쎈깨슬홀믿몽쏭핫펌볼륨캡굿짱끝뽈쑝뚝딱짜엣병숲널몰써덴던뭄죽궁낵긴튬웃뭉텀겐펍댁쭌칭멜벤처땅랩붙검킴솥엉범윈쟁낚샨똘킹격렁된롤적뇽썰잡를퀸몬넷존넘액윌뇨셉농펫팰며튀압뢰웨덤쉽받싼런팩욱밸밭져썸맹뱃쁨윗옴므령돔맘틀럭륜말셋뷔논읽닐험융려떤척닷밝텍턱룻짚넌첨씬절풋묘곳납왓옵픽퀘캠질렐톡젼헛넥흰뱅쁠띠깐뎃탠밌칸빅톤팝률옻벧확듀덮퉁섬딘즌튼넝꼼캣론럼쉴측젝쿱흡맑잼팀엑렛웹죠딕곁잎좌컵퀵떳쾌높딧벡놔쫌얄렙윙쳐변뎀즘랭없렬넣흔톨녹맞춤굴및움뚱잭솟밧픈찰슨롭삽녕붕슐쿤쟈맴뽁켐솜팜긋숨젬억졸셨첫뒤될셜찹빔됩챌꽁먼뚜펙킥뒷깜빡헌폐캬뽑삘툴겨땡떙럿겁렘냄곧셔흐균힌즙첸많념쩡칡징딴햄뼈웅춧퇴벚킬뻐썬츄촬옳껍벗앰묵찾뵙겠랫폼샾램렵숴봐갓넓떠앞뚫닙켄되싯왔씻잘섯략않것꼰혹긍답짧폴켑랐칙핵폭혁왜묻냐엇떻괴했겅짓삶겔덩훙촛듯쓴옮멧큘싫씩괜찮죄혐걱낳았뭐둠퀴쓸끈었렇겼꿔멈좁뭔딜잃봤앓팍햬헥귝얀듦같씀뿌엽훔걷깊휼엿냈쁜놓밖큼낫닌맙봇닫둔뀐캇듣붓랍켜숫잊혀틴튤못벼떨긷뛰슴찢맺줍쪽붉첩늙귓괘갔헝섹쥐낮챔캉섭쯤좀줘졌딥캘옙걀둘덟씹헸웠볍였픔셰잖낸핍냠넙뻔콕엮얻솝챙랬펀뮈셸쉿뻬흘꽂칵몇툰깥꼭낯탉놈빚늑줌첼삿뮌찐햅펭귄빽옹댓맷퀀잌끗꽈넛숩숀슥쉘룽싹륭훼왼흙똥갱겟콧뮬헐룰믈푼샛벳쑈퐁읍쿼벅앨뫼뗌롶껑웬삐톱값딤쁄훌젓펄줏쏙짙텝볕궐렴펑뉘탱륙꿉꾹얌떴쿰늄밑숑킷돗팻귤뀨욜뤼셴똑꺽웜꿍쿵돝깡쥴밋겸뱍띵쪼쌩갯젊텃헉랏굼랴똔깻긱팟휠덱찍멤깍븟멕쑥뺑쟝궃촉샴쏨녘갸윰픙뱀떄벙늪뿔뿍껏숖홉듸숟븍셈썅묭밈쯔맟캄낌탤쓱릎섞찡굯쏜솣쫀꼽굉돋묶씰힙쥰삥챠땐껴햐탬쭉횡촨뽄옾땀곶굄멩껌챈꾀켠룬잣텬웁깅윔낼쑤툼륵둣얏녜돛풉챨쭝엥깎욘뙈갭녁쪄훗웍덜뺴듭튠쮸얹쉔뻥괸씸쏀왈쨈섀옐갖뢍괄툇짠댑헹캅뀀껄휀똣넋빳틔댐띄헵꽐썽쳇눌쎌떢렷롸넉힝뿐얜쑨챤뵈볏묾쩌깽쟌뿜랠칫죤뺀캥댥퀄렝뭇챕칩뺭뽐맬샥냇퇘왠켘쌉횃댜섶몀큽펴얇슌슁끊엎슘졍돐숄넵톳팸젯빻붎꼐셩읜햔꿩댈딪늬빤잽깝볽됨쏴뛴컹땍틉끽꺠앉뤠릅쯴늠웸짤뻑뼝쫑쩍갬뗑앱뜌뜜쩜쌔섧잗뗴넨쨍껀릐눕뷜낄푹슟뭘꼈낀뚤뜬뜯삑냅뀝횰쩨쎔쌓놋샆췬힉싀헙앳껨덧윷겜놉싣뎁캌윅팁펏났옌펼핯훨썩뫔궂잿촘췄뚠땁얘튈깁봅짭벵킵넴빱엡쇳쉰딍띤삔갠콥쿳싁럷툽틍젖', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--tp', type=str, default='vert', help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)
