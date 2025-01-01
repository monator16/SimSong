#%% Import required modules
import os
import json
import csv

# JSON 파일이 있는 디렉토리 경로
json_directory = r"C:\Users\user1\Desktop\3-2\딥러닝\project\dataset\lastfm_subset\lastfm_subset"  # JSON 파일이 위치한 디렉토리 경로를 여기에 입력하세요.

# 결과를 저장할 딕셔너리
results = {}

# 하위 디렉토리를 포함하여 모든 JSON 파일 읽기
for root, dirs, files in os.walk(json_directory):
    for filename in files:
        if filename.endswith(".json"):  # JSON 파일 필터링
            filepath = os.path.join(root, filename)  # 파일 전체 경로
            
            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    data = json.load(file)  # JSON 파일 읽기
                    
                    # similars 필드에서 유사도 리스트 추출
                    similars = data.get("similars", [])
                    
                    # 파일 이름을 키로 저장 (확장자 제거)
                    file_key = os.path.splitext(filename)[0]
                    
                    # similars 리스트와 개수를 딕셔너리에 저장
                    results[file_key] = {"similarity_list": similars, "count": len(similars)}
            
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")

# 결과 확인
for key, value in results.items():
    print(f"File: {key}, Similarity Count: {value['count']}, Similarity: {value['similarity_list']}")  # 일부 데이터 출력
    
    
#%%

import csv

# CSV 파일 저장 경로
output_csv_path = "output.csv"

# CSV 파일 생성
with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    
    # 헤더 작성
    writer.writerow(["key", "id", "similarity_score"])
    
    # results 딕셔너리 순회
    for key, value in results.items():
        similarity_list = value.get("similarity_list", [])
        if similarity_list:  # similarity_list가 비어 있지 않은 경우
            for item in similarity_list:
                if len(item) == 2:  # [id, 유사도 값] 구조만 처리
                    id_value, similarity_score = item
                    writer.writerow([key, id_value, similarity_score])
        else:  # similarity_list가 비어 있는 경우
            writer.writerow([key, None, None])  # None으로 빈 값 처리

print(f"CSV 파일이 생성되었습니다: {output_csv_path}")

# %%
import csv

# 파일 경로
output_csv_path = "output.csv"
similarity_csv_path = "similarity_results.csv"

# key와 Track ID를 저장할 리스트
output_keys = set()
track_ids = set()

# output.csv에서 key 읽기
with open(output_csv_path, mode="r", encoding="utf-8") as output_file:
    reader = csv.DictReader(output_file)
    for row in reader:
        key = row.get("key")
        if key:
            output_keys.add(key)

# similarity_Results.csv에서 Track ID 읽기
with open(similarity_csv_path, mode="r", encoding="utf-8") as similarity_file:
    reader = csv.DictReader(similarity_file)
    for row in reader:
        track_id = row.get("Track ID")
        if track_id:
            track_ids.add(track_id)

# 중복된 항목 찾기
duplicates = output_keys.intersection(track_ids)

# 결과 출력
print(f"중복된 항목 ({len(duplicates)}개): {duplicates}")

# %%
import pandas as pd
import matplotlib.pyplot as plt
output_csv_path = "output.csv"
output = pd.read_csv(output_csv_path)
sim_score = output['similarity_score']

plt.figure()
plt.hist(sim_score, range = [0,1], bins = 1000)

# %%
plt.figure()
plt.hist(sim_score, range = [0.1,1], bins = 1000)
# %% threshold보다 큰 id만 남김
# 유사도가 0.1보다 작은 데이터를 제거하는 함수
def filter_similarities(results, threshold=0.1):
    filtered_results = {}
    for key, value in results.items():
        similarity_list = value.get("similarity_list", [])
        # 유사도가 threshold 이상인 항목만 필터링
        filtered_similarity_list = [item for item in similarity_list if item[1] >= threshold]
        # 필터링 결과가 있는 경우에만 추가
        if filtered_similarity_list:
            filtered_results[key] = {"similarity_list": filtered_similarity_list}
    return filtered_results

# 필터링 실행
filtered_results = filter_similarities(results, threshold=0.1)

    #%% mp3 600개만 고려
mp3_list = ['TRAAFOY128F146CC17.mp3',
 'TRAAGJV128F1464090.mp3',
 'TRAAIHL128F92E6DDA.mp3',
 'TRAAIMC128F42625C6.mp3',
 'TRAAKTS128F429F622.mp3',
 'TRAAPPQ128F14961F5.mp3',
 'TRAAZGC12903CB5DFB.mp3',
 'TRAAZQW128F93430DC.mp3',
 'TRAABJV128F1460C49.mp3',
 'TRAADQX128F422B4CF.mp3',
 'TRABEIU128F4266CC2.mp3',
 'TRAJDTK128F429FFDF.mp3',
 'TRALLKT12903CF4912.mp3',
 'TRALLSG128F425A685.mp3',
 'TRALPRX128F92DC3CA.mp3',
 'TRALXLY128F148B35B.mp3',
 'TRALYVS12903CE92AC.mp3',
 'TRALZXZ128F92E26EF.mp3',
 'TRAMEBP12903CD7EEA.mp3',
 'TRAMXVM128F930BB2C.mp3',
 'TRANEGJ128F428617E.mp3',
 'TRANEMQ128F42A10A8.mp3',
 'TRANLHL12903CF88E5.mp3',
 'TRANNUZ128F145E45B.mp3',
 'TRANSYV128F428A00C.mp3',
 'TRANXQI128F4294976.mp3',
 'TRAJHDR128F92FA863.mp3',
 'TRAMKKJ128F1463612.mp3',
 'TRAMNKN12903D03F49.mp3',
 'TRANHGL128F145A6A6.mp3',
 'TRANPWV128F9341281.mp3',
 'TRAOALO128F427CB1D.mp3',
 'TRAOHJX128F92EF67A.mp3',
 'TRAOJCT128E0793F5E.mp3',
 'TRAOUOI128F4290E44.mp3',
 'TRAOTWV128F147731A.mp3',
 'TRAOYHS128F427EF8B.mp3',
 'TRAPCHT128F4239830.mp3',
 'TRAPKBI128F92F9D33.mp3',
 'TRAPDQG128F93018C0.mp3',
 'TRAPSHW128F14560A6.mp3',
 'TRAPYRX128F427F248.mp3',
 'TRAQEUQ128F425282E.mp3',
 'TRAQSWI128F9343039.mp3',
 'TRAQXKZ128F429D175.mp3',
 'TRARJEK128F930B3AA.mp3',
 'TRARKEH128F1463CB2.mp3',
 'TRARKLX128F934C646.mp3',
 'TRASIVA128F933DB52.mp3',
 'TRARPCK128F92FEA7F.mp3',
 'TRGGWEE128F92D5BCA.mp3',
 'TRGBGSA128F92F9387.mp3',
 'TRGBMSS128F4295043.mp3',
 'TRGBVDZ128EF356AF9.mp3',
 'TRGCYUP12903CF418D.mp3',
 'TRGEMSN128E07944A8.mp3',
 'TRGETHW128F93321E9.mp3',
 'TRGEVHJ128F146CC1E.mp3',
 'TRGFEUS128F148AEA3.mp3',
 'TRGGIJA128F14A4702.mp3',
 'TRGGNWS128F1463468.mp3',
 'TRGHDOJ128F1468444.mp3',
 'TRGHFQB128F14A3A80.mp3',
 'TRGHGLN128F145FE62.mp3',
 'TRGHVGY128F92F2254.mp3',
 'TRGIAOR128F429EF4D.mp3',
 'TRGILIS128F424ED92.mp3',
 'TRGIRLD128F92CBD6C.mp3',
 'TRGBFMF128F933801B.mp3',
 'TRGCXFB128F1459C9A.mp3',
 'TRASIYE128F92F9261.mp3',
 'TRASPAS128E078DFB3.mp3',
 'TRATFEN128F14A3A6E.mp3',
 'TRATGSL128F93423C2.mp3',
 'TRATRDG128F427313B.mp3',
 'TRAUEWN128F932C32C.mp3',
 'TRATJXC128F93124CE.mp3',
 'TRAUIHC128F427EEF9.mp3',
 'TRAUMNH128F92FB67F.mp3',
 'TRAUNPM12903CB7CCC.mp3',
 'TRJELBN128F42721CF.mp3',
 'TRJFDGD128EF340E06.mp3',
 'TRJFFFS128E078AADD.mp3',
 'TRJHALT128F933B0E9.mp3',
 'TRJHUAT128F425BA9C.mp3',
 'TRJEMHD128F4285513.mp3',
 'TRJGSJQ128F14AE2E4.mp3',
 'TRJEUFZ128F92F3FAE.mp3',
 'TRJFAOJ128F931D308.mp3',
 'TRJFZHE128F9336402.mp3',
 'TRJGBFY12903CF47AC.mp3',
 'TRJHCKZ128F427ECF0.mp3',
 'TRGIVGF128F92FE01C.mp3',
 'TRGIXTN12903D0194A.mp3',
 'TRGJFCC128F9306674.mp3',
 'TRGJMCB12903CF8AA9.mp3',
 'TRGJTIV128F9302640.mp3',
 'TRGJWNK128F9351E94.mp3',
 'TRGKCEN128F4228AC9.mp3',
 'TRGNVAM128F428F119.mp3',
 'TRGLNFK128F422305E.mp3',
 'TRGLPHZ128F9335B81.mp3',
 'TRGLURC128F145A6A3.mp3',
 'TRGNSQY128F148D76D.mp3',
 'TRCKIXM128F427EBBD.mp3',
 'TRCLKCO12903CDDB09.mp3',
 'TRCMZXV128E07943F4.mp3',
 'TRCQLJF128F425CDFB.mp3',
 'TRCIWCS128F931FB25.mp3',
 'TRCQTHS128F92F3D37.mp3',
 'TRCINUL128F42BCBE6.mp3',
 'TRCKOZR128F149169F.mp3',
 'TRCMTWH128F4251CEC.mp3',
 'TRCMXJZ12903CB2303.mp3',
 'TRCQOOC128F931B18F.mp3',
 'TRCHFMO128F4295137.mp3',
 'TRCHINK128F422B87A.mp3',
 'TRCHXXE128F428547C.mp3',
 'TRCIJHB128F42A7CF6.mp3',
 'TRCJDBW12903CE70C7.mp3',
 'TRCIXBB128F146370D.mp3',
 'TRCMJAO128F426D477.mp3',
 'TRCMTOK128F92DEF8E.mp3',
 'TRCLZLM128F42370C4.mp3',
 'TRCQUSU128E0792C34.mp3',
 'TRCRCHG128E07837CE.mp3',
 'TRCRFXV12903CEC3EF.mp3',
 'TRCRXDJ128F9343EEE.mp3',
 'TRCSXPY12903CDF897.mp3',
 'TRCSIDQ128F4260B2E.mp3',
 'TRCTBQN128E0792C3B.mp3',
 'TRCTJER12903CB76E9.mp3',
 'TRCUKHR128F92DF656.mp3',
 'TRCUNSN128E0781DC1.mp3',
 'TRCUVMP128F4267FAA.mp3',
 'TRCVPIG128F426FFB7.mp3',
 'TRCVSDY12903CFD7C6.mp3',
 'TRCVUUV128F42482AE.mp3',
 'TRCWETL128F42507C3.mp3',
 'TRBYXHQ128F427F245.mp3',
 'TRBYMXP128F425A348.mp3',
 'TRBYWIS128F428C79B.mp3',
 'TRBZBLX128F933AD74.mp3',
 'TRZGAPY128F1477025.mp3',
 'TRZGFEV128F427C38B.mp3',
 'TRZGHKC128F14604C5.mp3',
 'TRDQCIJ128F423D24D.mp3',
 'TRCAXUT128E078E5DF.mp3',
 'TRCAYPK128F92E7190.mp3',
 'TRCBPZQ128F92E26F4.mp3',
 'TRCBSQP12903CF4DAA.mp3',
 'TRCCIZB128C7196773.mp3',
 'TRCCNIG128F92F56A1.mp3',
 'TRCDPAG128F42801B8.mp3',
 'TRCEIPR128F92DE23B.mp3',
 'TRCEKLJ12903CB6ACE.mp3',
 'TRCCYGO128F933DAEB.mp3',
 'TRCDOOE128F42AA865.mp3',
 'TRBYPCJ128F146715F.mp3',
 'TRCFOSM128F428B5A4.mp3',
 'TRCGTGE128F92F06B9.mp3',
 'TRHBJFV128F92D43C3.mp3',
 'TRHCRPS128F4262B64.mp3',
 'TRGZIVY128F4268098.mp3',
 'TRHABQN128E07932E8.mp3',
 'TRHEGQZ128F92E4B29.mp3',
 'TRHEMBZ128F4271E26.mp3',
 'TRHFAHH128F429556B.mp3',
 'TRHFAYO12903CF0BDF.mp3',
 'TRHFPML128E0793522.mp3',
 'TRHGXAN12903CBD07B.mp3',
 'TRHGCAP128F146D5D0.mp3',
 'TRHHWDM128F429EE4A.mp3',
 'TRHIRCR128F930BDD9.mp3',
 'TRHISSO128F930B56C.mp3',
 'TRHJHEG12903D0349C.mp3',
 'TRHJZFU128E0793F5A.mp3',
 'TRHGMJK128F92D806F.mp3',
 'TRHEBCH128F92F74AC.mp3',
 'TRHGCKO12903D15317.mp3',
 'TRHLGZQ128F933B0DF.mp3',
 'TRHLIDH128F426150F.mp3',
 'TRHLUCU128F145834C.mp3',
 'TRHLOCM128F4233ADF.mp3',
 'TRHLZKD128C71965C6.mp3',
 'TRHMIPY128F93373D0.mp3',
 'TRHMDYA128F42BAD0F.mp3',
 'TRHMGOP128F4295022.mp3',
 'TRHMBSZ128F92D2B67.mp3',
 'TRHMKVZ128F42AA3A4.mp3',
 'TRHMVVD128F9355699.mp3',
 'TRHMXVQ128F92F37E1.mp3',
 'TRHNVYP128F4262C22.mp3',
 'TRHNXTZ128F4271F9A.mp3',
 'TRHOFUK128F42650EF.mp3',
 'TRHOHFZ12903CA8562.mp3',
 'TRHOPPY128E078E5E4.mp3',
 'TRHQBNG128F9300881.mp3',
 'TRHQFER12903CAC0CF.mp3',
 'TRHRBGY128F1466076.mp3',
 'TRYDNGV12903CD9FBD.mp3',
 'TRYFAMN128F428D7C0.mp3',
 'TRYFHTO128F9319BA8.mp3',
 'TRYFIOW128F4220E7F.mp3',
 'TRYFZZT128E0790AD2.mp3',
 'TRYGLIM128F425A36F.mp3',
 'TRYGPKI128F932C458.mp3',
 'TRYGQAF128F92F5BFC.mp3',
 'TRYGTYN128F92F7442.mp3',
 'TRYHEOV128F14A3AA0.mp3',
 'TRYHRYO128F1462855.mp3',
 'TRYISLM128E07818D9.mp3',
 'TRYCDPU128F92F0127.mp3',
 'TRYIBCM128F428EE79.mp3',
 'TRYIMOJ128F93151B5.mp3',
 'TRYGFMO128F42833D2.mp3',
 'TRYECMI128F4239B06.mp3',
 'TRYDCFS128F93016F7.mp3',
 'TRYEXPK128F42ACF07.mp3',
 'TRYIWAS128F42790DE.mp3',
 'TRYMAND128F14A3A50.mp3',
 'TRYJCWB128E0784883.mp3',
 'TRYJGQU128F9310F86.mp3',
 'TRYKEIX128F4278959.mp3',
 'TRYLNRT128F4215A95.mp3',
 'TRYLSBR12903CC09C5.mp3',
 'TRYKFIH128F4265A1F.mp3',
 'TRYMWAD128F4271AFF.mp3',
 'TRYNRPX128F425C350.mp3',
 'TRYOPQE128F92DDED2.mp3',
 'TRYNHSE12903CBF477.mp3',
 'TRYPQFX12903CB819C.mp3',
 'TRYQRAZ128F149E6AA.mp3',
 'TRYQXBP128E0791309.mp3',
 'TRYQZOD128E0781D8F.mp3',
 'TRYRZFF128F933BBBB.mp3',
 'TRYSHGM128F4244A75.mp3',
 'TRYSJDG128F14560AE.mp3',
 'TRYTKLB128F92F11FE.mp3',
 'TRBMVLL128E0791BC0.mp3',
 'TRBMZSW128F932312A.mp3',
 'TRBOEUP128F421C08F.mp3',
 'TRBPGZR128F145A383.mp3',
 'TRBPMBC128F932D478.mp3',
 'TRBPUSI128F931B175.mp3',
 'TRBPXND128F42873D6.mp3',
 'TRBRPIA128F4289506.mp3',
 'TRBRSNN128F4284E8C.mp3',
 'TRBSIHA128F145A7E7.mp3',
 'TRBTGRS128F1459BBF.mp3',
 'TRBSCAX128F42B026D.mp3',
 'TRBTVII12903CFD7C4.mp3',
 'TRBUAJF128F425E5AB.mp3',
 'TRBUMFX128F14AE2E0.mp3',
 'TRBVLDI12903CD3AC6.mp3',
 'TRBSIZJ12903CB115C.mp3',
 'TRBUTEJ128F14AE213.mp3',
 'TRBTRXZ128F145E4D8.mp3',
 'TRBVCYN128F92F3487.mp3',
 'TRYTKNG128F1495BD0.mp3',
 'TRYTQHU128F4260AF5.mp3',
 'TRYUHIC128F930C9BF.mp3',
 'TRYUSQL128F42825EA.mp3',
 'TRYUWZC128F42550F2.mp3',
 'TRYUQYF128F4270023.mp3',
 'TRYVCYM128F92ECCFC.mp3',
 'TRYVDLP128F422B47D.mp3',
 'TRYVMLP128F42AAA6C.mp3',
 'TRYVRMC128F92F3C9C.mp3',
 'TRYVXLY128F1456051.mp3',
 'TRYWCRP128F426BFFF.mp3',
 'TRYWIRR12903CE7C93.mp3',
 'TRYWMMX12903CD4BED.mp3',
 'TRYWTQS12903CB4BE4.mp3',
 'TRYYCGL12903CB11C2.mp3',
 'TRYYDBK128F930FDBC.mp3',
 'TRYYJQW12903CECBC9.mp3',
 'TRYYPVU12903CCE5A6.mp3',
 'TRBVMBC128F932E6A5.mp3',
 'TRBVWFH128F92F6BFB.mp3',
 'TRYYTVV12903D03F75.mp3',
 'TRDHAMO128F93164AD.mp3',
 'TRDHEGD128F145EC37.mp3',
 'TRDHKJT12903CC1D3C.mp3',
 'TRDHRYP128F427906F.mp3',
 'TRDLVIC128F933B0E7.mp3',
 'TRDMZYI12903CF3A27.mp3',
 'TRDIKNI128F92C403C.mp3',
 'TRDIMTT128F428C8DC.mp3',
 'TRDJGCL128F4265570.mp3',
 'TRDJGCT128F93002A4.mp3',
 'TRDJIPJ128F9321317.mp3',
 'TRDKFYE12903CEC609.mp3',
 'TRDOBBS12903CF4189.mp3',
 'TRDOFRX128F92CD19F.mp3',
 'TRDOWXI128E078EFAD.mp3',
 'TRDPBEW128F4290718.mp3',
 'TRDMASQ128F145BFA3.mp3',
 'TRDMJIF128F42960BB.mp3',
 'TRDJNVA128F427106A.mp3',
 'TRDKHAZ128F429821F.mp3',
 'TRYZJUW128F934A95E.mp3',
 'TRBWVQJ128F42624AD.mp3',
 'TRBXHXP12903CD78E7.mp3',
 'TRBXLFU128F427A494.mp3',
 'TRBWFIV128F9331289.mp3',
 'TRBWUFL128F42ACEE2.mp3',
 'TRBXLIP128F4277241.mp3',
 'TRBWQXE128F92E26F6.mp3',
 'TRBYKWX128F4263C88.mp3',
 'TRZARES128F9351D52.mp3',
 'TRZAWBV128F92FE646.mp3',
 'TRZBFFQ128F14AE050.mp3',
 'TRZCCPB128F1480D66.mp3',
 'TRZCLXJ128F429CF49.mp3',
 'TRZCRBN12903CA6847.mp3',
 'TRZDQRG128E07933BE.mp3',
 'TRZDZIR12903CBB923.mp3',
 'TREOKFT128F146129A.mp3',
 'TREOSRI128F42B2FE5.mp3',
 'TREPHOD128F92D5435.mp3',
 'TREPHRP128F1484C76.mp3',
 'TREPTAO128F428C2A3.mp3',
 'TREPWBR128F930FB68.mp3',
 'TREQIYQ128F92E6AE6.mp3',
 'TREQPPW128F148778C.mp3',
 'TREQWFI128F92CAE57.mp3',
 'TREQYKG128F427E7A1.mp3',
 'TRESIZK128F9331F4D.mp3',
 'TRESKED128F92FC0D5.mp3',
 'TRESMGD128EF350312.mp3',
 'TRESMLC128F425A385.mp3',
 'TRESRKO12903CAFA55.mp3',
 'TRETBCK128F92FEA76.mp3',
 'TRETPUT128F427E11E.mp3',
 'TRETRUQ128F9347FC1.mp3',
 'TREUGEP128F145B795.mp3',
 'TRZETWI128F93089E3.mp3',
 'TREUITQ12903D05934.mp3',
 'TRZFDZJ12903D152E3.mp3',
 'TRZFKTR128F42B06C6.mp3',
 'TRZFUWI128F92DF7FB.mp3',
 'TRDPELC128F92CFE45.mp3',
 'TRDPKPN128F146370F.mp3',
 'TRDQHBR128F42770D2.mp3',
 'TRDRDOD128F429AB2F.mp3',
 'TRDRFLI128F92C94CB.mp3',
 'TRDQWDZ128F933BBBF.mp3',
 'TRDSFYQ128F4222EC9.mp3',
 'TRZGJEU128E0780F36.mp3',
 'TRZGXCG128F9312BFD.mp3',
 'TRZHBLJ128F92E1540.mp3',
 'TRDSHMN128E078F1D1.mp3',
 'TRDSRDJ128F930742A.mp3',
 'TRDURQN128F14961F8.mp3',
 'TRDVMFM128F424CD4C.mp3',
 'TRDTISO128F4268A5E.mp3',
 'TRDTQJI128F92E37A7.mp3',
 'TRDVWQT128F4289D8E.mp3',
 'TRDVWVL128F42655EA.mp3',
 'TRDVYGV128F930B50B.mp3',
 'TRDWGVF12903D09518.mp3',
 'TRDWTEB128F1464BE3.mp3',
 'TRDXCXL12903CF4184.mp3',
 'TRZIOXF128F427264D.mp3',
 'TRZIVQY12903CEA102.mp3',
 'TRZIXZE128F426F410.mp3',
 'TRZKBNO12903CEB906.mp3',
 'TRBZJUP128EF356B05.mp3',
 'TRBZNMB128F9307828.mp3',
 'TRCAMRS128E078E034.mp3',
 'TRCABEE128F145864C.mp3',
 'TRZKWIV128F931224B.mp3',
 'TRZKDET128F1459B0E.mp3',
 'TRZKOPA128F92FFB93.mp3',
 'TRDXRZV128F1488829.mp3',
 'TRZKYSN128F9330813.mp3',
 'TRNDFSR128F423E59D.mp3',
 'TREUNLR128F422B481.mp3',
 'TREUQIH128F149DA08.mp3',
 'TREUXNQ12903CA6267.mp3',
 'TREUWLS128F92FA3D1.mp3',
 'TRDXVWB128F42AA169.mp3',
 'TRFHRIY128F42A04F7.mp3',
 'TRFKEZE128F4259782.mp3',
 'TRFMYVK128F4289C6D.mp3',
 'TRFKEVF12903CECBD3.mp3',
 'TRFKJVK128F426E70D.mp3',
 'TRFGELX128C71965C8.mp3',
 'TRFGQVP128F93321F7.mp3',
 'TRFGRSH128F42965DE.mp3',
 'TRFJLQK128F92F9034.mp3',
 'TRFJWYF128F14AE217.mp3',
 'TRNBGVB128F426AC12.mp3',
 'TRFFGIH128F1459C35.mp3',
 'TRFFIFC128F92D1EC1.mp3',
 'TRFLEKF12903C9DD69.mp3',
 'TRFMMOG128F92CBD5F.mp3',
 'TRFNDQP128F428C298.mp3',
 'TRFNKCM128F426B279.mp3',
 'TRFGVEO128F42636DF.mp3',
 'TRFHBLD128F92CAE4F.mp3',
 'TRFJEJF128F4265BBE.mp3',
 'TRFKHLB128F9320174.mp3',
 'TRNCYSP128F426DF26.mp3',
 'TRNDFFB128F93496EE.mp3',
 'TRZNONX12903CEA3DC.mp3',
 'TRNDFGY128F148D04D.mp3',
 'TRZODUN128F42440AB.mp3',
 'TRDYCUI128F92D5BC3.mp3',
 'TRDYICR128F421A600.mp3',
 'TRDYWSL128F4267D45.mp3',
 'TRDYSYQ128F42934BA.mp3',
 'TRDYYDZ128F93353F7.mp3',
 'TRZOKKF128F145EC4C.mp3',
 'TRZOLOQ128F428FA00.mp3',
 'TRZPIPO128F931B2BA.mp3',
 'TRZQCMI128F42A8DC8.mp3',
 'TRZQPVP128F92FB5FA.mp3',
 'TRZRBRV128F42AD2DD.mp3',
 'TRZRJPO128F92EF9C8.mp3',
 'TRZSIGV128F931B189.mp3',
 'TRFPLHY128F9302C67.mp3',
 'TRFPSHD128F92FE6CE.mp3',
 'TRFQPMZ128E07948DF.mp3',
 'TRFQRYC12903CD0BB9.mp3',
 'TRFRNAW128F14672DB.mp3',
 'TRFRASA128F9306533.mp3',
 'TRFROOK12903D0F9CA.mp3',
 'TRZSXBS128F4268C9C.mp3',
 'TRZTNOJ128F42627DE.mp3',
 'TRZUCWY128EF34B6C7.mp3',
 'TRZUQEM128F92E6AE7.mp3',
 'TRZUWOD128F145EC17.mp3',
 'TRZVLLV128F4288445.mp3',
 'TRZVIDL128F931B19E.mp3',
 'TRZWJBI128F1487791.mp3',
 'TRZWOBY128E079124D.mp3',
 'TRZWONJ128F92F22C0.mp3',
 'TRZYGHF128F429FCF5.mp3',
 'TRZYPVV12903CBE29D.mp3',
 'TRZZSJM128F42481C5.mp3',
 'TRZZVWE128F1477584.mp3',
 'TRIMTDE128F92F740A.mp3',
 'TRINAXZ128F931A8B8.mp3',
 'TRIPTUT128F428F9FB.mp3',
 'TRIQFGY128F425BFEB.mp3',
 'TRIQPMR128F149C308.mp3',
 'TRIQTYZ128F422B896.mp3',
 'TRIRCBJ12903CF47B4.mp3',
 'TRIRMPK12903CF6F93.mp3',
 'TRISGTK128F428F9F8.mp3',
 'TRISMIC128F42739CE.mp3',
 'TRITUZT128F4277742.mp3',
 'TRIUIVR12903D16876.mp3',
 'TRIUTZJ128C7196BD6.mp3',
 'TRIVMAY128F4276E15.mp3',
 'TRIWIXJ128F42782F5.mp3',
 'TRIWRJS128F92F6614.mp3',
 'TRIMNLQ128F92F10E4.mp3',
 'TRIMGYM128E07848C0.mp3',
 'TRIUJEC12903CBCD99.mp3',
 'TRIVRGN128F9324F50.mp3',
 'TRIXJSX128E0791734.mp3',
 'TRLADFP12903CA5021.mp3',
 'TRLAOFB128F423B456.mp3',
 'TRKZZZF128F1491695.mp3',
 'TRLAKII128F9341286.mp3',
 'TRKVMKN128F14604C6.mp3',
 'TRLAFGW128F4259B24.mp3',
 'TRLBRWR12903CE128C.mp3',
 'TRLEPSG128F92C2500.mp3',
 'TRKVQOB128F145ABE0.mp3',
 'TRKWAMY128F4294890.mp3',
 'TRKWOLH128F4239803.mp3',
 'TRKXOQJ128F42803AE.mp3',
 'TRKZDJX128F148C634.mp3',
 'TRLBWDV128F92D8274.mp3',
 'TRLCDNH12903CEBF21.mp3',
 'TRLFMCD128F4238837.mp3',
 'TRKZMVI128C7196770.mp3',
 'TRKVRKB128F93371E3.mp3',
 'TRLDWAF128F4285917.mp3',
 'TRLFTCX128F425D445.mp3',
 'TRLFWMC128F9317C9E.mp3',
 'TRLGSOU128F42960C5.mp3',
 'TRLGVOV128F146A19F.mp3',
 'TRLGVTV128F92E26F0.mp3',
 'TRLGZXI128F146346C.mp3',
 'TRLHKOV128F9341283.mp3',
 'TRLHNLY12903CC0587.mp3',
 'TRLHOZK128F934314C.mp3',
 'TRTEWBX128F42599A1.mp3',
 'TRTFWXE128F4260512.mp3',
 'TRONVAO128F4239AFF.mp3',
 'TROOWWY128F14560A8.mp3',
 'TRSJMDN128E078109F.mp3',
 'TRNCIOP128F426A93D.mp3',
 'TRNDDSD128F92E1AE7.mp3',
 'TROIRNZ12903CE5D91.mp3',
 'TROJRYI128F425A79A.mp3',
 'TROKNNM128F92F3D5B.mp3',
 'TROJTQV128F145AFFB.mp3',
 'TRONAWX128F93351E3.mp3',
 'TRONONO128F9302C5A.mp3',
 'TRTAGZQ128F4227316.mp3',
 'TRTAVUY128F145A647.mp3',
 'TRTBZHX12903CB018E.mp3',
 'TRTEPMU12903CF4301.mp3',
 'TRPKKQK128F92F36AE.mp3',
 'TRPOCKW128F427F136.mp3',
 'TRPOQDZ128F92D5BC0.mp3',
 'TRPKZZT128F92FE049.mp3',
 'TRPKMIF128F93210DC.mp3',
 'TRPKOZK128F92C503A.mp3',
 'TRPLENJ128F425A342.mp3',
 'TRPLZCJ128F92FFB1A.mp3',
 'TRPMHHP128F933DB50.mp3',
 'TRPMHRE128F428AA0F.mp3',
 'TRPMJHA128F429D16E.mp3',
 'TRPMJHJ128F4272263.mp3',
 'TRPPDKG128F145AFF2.mp3',
 'TRPMWXN128E0792C24.mp3',
 'TRPLJTF128F429D126.mp3',
 'TRPMPUW128E0784312.mp3',
 'TRPPLDH12903CBAF23.mp3',
 'TRSKEFS128F931F0D8.mp3',
 'TRSJZDC12903CE6DB1.mp3',
 'TRONWPP128F42A001A.mp3',
 'TRSJSXO128F428CAE1.mp3',
 'TRSMHGE128F4259C72.mp3',
 'TRSKOEL128F932C5DB.mp3',
 'TRSLCDZ128E07847D6.mp3',
 'TRSMNNR128F93088DB.mp3',
 'TRSMTDW128F42ACA8E.mp3',
 'TRSLWEY128F145DCA1.mp3',
 'TRSMZIG128F428253E.mp3',
 'TRSNCDZ128F92DE641.mp3',
 'TRSNDBU128F4271E6E.mp3',
 'TRSMWQW128F428CD91.mp3',
 'TRSOITY128F1496251.mp3',
 'TRSKJLH128F4290A53.mp3',
 'TRSJRFK128F427F24C.mp3',
 'TRJSROY12903C9D802.mp3',
 'TRJSXPR128F147D6B3.mp3',
 'TRJTZIZ12903CBDDD1.mp3',
 'TRJQYQI128F421CAFA.mp3',
 'TRJRGLQ128F426ABA2.mp3',
 'TRJRLLO128F428450A.mp3',
 'TRJSIYE128F93278E4.mp3',
 'TRJTZXI128F4289C71.mp3',
 'TRJSLGC128F42B84FD.mp3',
 'TRJUAGI128F426177C.mp3',
 'TRJUASI128F423E59A.mp3',
 'TRVBXDW128F427AF96.mp3',
 'TRVBXXY128F9352B12.mp3',
 'TRVCCFV128F149D9C6.mp3',
 'TRVDFFJ128F92F3C97.mp3',
 'TRVHNDP128F4223058.mp3',
 'TRVHMJJ128F425F693.mp3',
 'TRVHWEM128EF365F44.mp3',
 'TRUZUIN128F4289AD1.mp3',
 'TRVADLK128F931D596.mp3',
 'TRVBGMW12903CBB920.mp3',
 'TRVBSZC128F92FE5A7.mp3',
 'TRVBXQZ128F4286D27.mp3',
 'TRVCSWO128F1465A30.mp3',
 'TRVDZVR128E0792C3E.mp3',
 'TRVDXEY128F930170D.mp3',
 'TRVEVQO128F425282A.mp3',
 'TRVEJTZ12903CA367C.mp3',
 'TRVHFMV128F145C0BD.mp3',
 'TRVHIJI12903CED6C2.mp3',
 'TRVAUUN128F422B48E.mp3',
 'TROGMWB128F42482A9.mp3',
 'TRVIHJW128E0792C28.mp3',
 'TRVISOX128F427F3AC.mp3',
 'TRSEKVZ12903CBD4AE.mp3',
 'TRSFQRG128F147A888.mp3',
 'TRSGGBD128F145B163.mp3',
 'TRSHRXC128E078FCB0.mp3',
 'TRSHROS128F14AE210.mp3',
 'TRSHVFG128F92E6860.mp3',
 'TRSHXOI128F146B1AE.mp3',
 'TRSFOPX128F4261B18.mp3',
 'TRSHOVL128F42636EB.mp3',
 'TRSDWDN128F4274C03.mp3',
 'TRSJLFA128F92F661B.mp3',
 'TROHCDR128F422B4D0.mp3',
 'TROHQAK128F424DFC7.mp3',
 'TROIFBI128F425976A.mp3',
 'TROMEWO128F9310EAA.mp3',
 'TROKBBC12903CE7C98.mp3',
 'TROIWMZ12903CECECA.mp3',
 'TROMGOI128F93124AA.mp3',
 'TROMZRT128F146A640.mp3',
 'TRTAUAR128F428E63E.mp3',
 'TRTBARA12903CBCB70.mp3',
 'TRTCLFS128E0799AD4.mp3']


# %% 위의 filtered_results를 기반으로 정리된 csv 만들기
def index_sets_by_id(filtered_results):
    # 집합 리스트와 ID-집합 매칭 결과를 저장할 딕셔너리
    sets = []
    id_to_sets = {}

    # 1. 각 key와 연결된 ID를 하나의 집합으로 저장
    for key, value in filtered_results.items():
        similarity_list = value.get("similarity_list", [])
        ids = {key}  # key 포함
        ids.update(item[0] for item in similarity_list)  # 리스트 내 id 추가
        sets.append(ids)
    
    # 2. 집합 정리: 교집합이 있는 집합을 하나로 합침
    merged_sets = []
    while sets:
        current_set = sets.pop(0)
        overlaps = [s for s in sets if not s.isdisjoint(current_set)]  # 교집합 있는 집합 찾기
        for overlap in overlaps:
            current_set.update(overlap)  # 현재 집합에 교집합 있는 집합 병합
            sets.remove(overlap)  # 병합된 집합 제거
        merged_sets.append(current_set)

    merged_sets = sets
    # 3. 각 집합에 인덱스 부여
    for idx, group in enumerate(merged_sets):
        for item in group:
            if item not in id_to_sets:
                id_to_sets[item] = []
            id_to_sets[item].append(idx)  # ID가 속한 집합의 인덱스 추가

    return id_to_sets, merged_sets

# 실행
id_to_sets, merged_sets = index_sets_by_id(filtered_results)


# 결과 출력
print("각 ID가 속한 집합의 인덱스:")
for id_, indices in id_to_sets.items():
    print(f"{id_} : 집합 {', '.join(map(str, indices))}")

print("\n모든 집합:")
for idx, group in enumerate(merged_sets):
    print(f"집합 {idx}: {group}")
    


#%%
def index_sets_by_id(filtered_results, min_overlap_size=2):
    sets = []
    id_to_sets = {}

    # 1. 각 key와 연결된 ID를 하나의 집합으로 저장
    for key, value in filtered_results.items():
        similarity_list = value.get("similarity_list", [])
        ids = {key}
        ids.update(item[0] for item in similarity_list)
        sets.append(ids)

    # 2. 집합 정리: 교집합 크기 조건 추가
    merged_sets = []

    while sets:
        current_set = sets.pop(0)
        overlaps = [s for s in sets if len(current_set & s) >= min_overlap_size]
        for overlap in overlaps:
            current_set.update(overlap)
            sets.remove(overlap)
        merged_sets.append(current_set)

    # 병합 이후 남은 집합 추가
    merged_sets.extend(sets)

    # 3. 각 집합에 인덱스 부여
    for idx, group in enumerate(merged_sets):
        for item in group:
            if item in mp3_list:
                if item not in id_to_sets:
                    id_to_sets[item] = []
                id_to_sets[item].append(idx)

    return id_to_sets, merged_sets


id_to_sets, merged_sets = index_sets_by_id(filtered_results)

#%%
def index_sets_by_id(filtered_results, mp3_list, min_overlap_size=2):
    sets = []
    id_to_sets = {}

    # 1. 각 key와 연결된 ID를 하나의 집합으로 저장
    for key, value in filtered_results.items():
        similarity_list = value.get("similarity_list", [])
        ids = {key}
        ids.update(item[0] for item in similarity_list)
        sets.append(ids)

    # 2. 집합 정리: 교집합 크기 조건 추가
    merged_sets = []

    while sets:
        current_set = sets.pop(0)
        overlaps = [s for s in sets if len(current_set & s) >= min_overlap_size]
        for overlap in overlaps:
            current_set.update(overlap)
            sets.remove(overlap)
        merged_sets.append(current_set)

    # 병합 이후 남은 집합 추가
    merged_sets.extend(sets)

    # 3. 'mp3_list'에 포함된 ID만 남기고 필터링
    filtered_merged_sets = []
    for group in merged_sets:
        filtered_group = {item for item in group if item+'.mp3' in mp3_list}
        if filtered_group:  # 비어 있지 않은 집합만 추가
            filtered_merged_sets.append(filtered_group)

    # 4. 각 집합에 인덱스 부여
    for idx, group in enumerate(filtered_merged_sets):
        for item in group:
            if item not in id_to_sets:
                id_to_sets[item] = []
            id_to_sets[item].append(idx)

    return id_to_sets, filtered_merged_sets

id_to_sets, merged_sets = index_sets_by_id(filtered_results, mp3_list=mp3_list)

#%%
import csv

# input CSV 파일 경로
input_csv_file = 'filtered_lyrics_file.csv'
# output CSV 파일 경로
output_csv_file = pd.read_csv('final_lyrics_with_sets.csv')

# filtered_lyrics_file.csv 불러오기
with open(input_csv_file, mode="r", encoding="utf-8") as infile, open(output_csv_file, mode="w", encoding="utf-8", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    header = next(reader)  # 첫 번째 행(헤더) 읽기
    header.append("Set Index")  # 마지막 열에 'Set Index' 추가
    writer.writerow(header)
    
    for row in reader:
        track_id = row[0]
        
        # Track ID가 id_to_sets에 존재하는 경우
        if track_id in id_to_sets:
            set_indices = id_to_sets[track_id]
            row.append(", ".join(map(str, set_indices)))  # 집합 인덱스 추가
            writer.writerow(row)  # 해당 행을 output CSV에 추가

#%% 위에서 만든 'filtered_lyrics_with_sets.csv' 분석
import pandas as pd
import matplotlib.pyplot as plt

# 'filtered_lyrics_with_sets.csv' 파일 불러오기
output_csv_file = pd.read_csv('final_lyrics_with_sets.csv')
result = pd.read_csv(output_csv_file)

# "Set Index" 컬럼에서 집합 인덱스 정보 추출
set_index_column = result["Set Index"]

# 각 노래가 몇 개의 집합에 속하는지 카운트
# 각 노래가 속한 집합 인덱스를 분리하여 각 노래가 속한 집합의 수를 세기
song_set_count = set_index_column.str.split(",").apply(lambda x: len(set(x)))  # 집합의 중복을 제거하고 카운트

# 각 노래의 집합 개수 히스토그램
plt.figure(figsize=(10, 6))
plt.hist(song_set_count, bins=range(1, song_set_count.max() + 2), edgecolor='black', alpha=0.7)
plt.title('Distribution of Set Membership per Song')
plt.xlabel('Number of Sets per Song')
plt.ylabel('Frequency')

# x축을 정수로 설정
plt.xticks(range(1, song_set_count.max() + 1))
plt.show()

#%% 집합이 총 몇개인지 확인

# 1. 집합 인덱스를 모두 나열 (쉼표로 구분된 값들에서 각 집합 인덱스만 추출)
set_indices = set_index_column.str.split(",").explode().astype(int)

# 2. 제일 큰 집합 인덱스 값 구하기
max_set_index = set_indices.max()
print(max_set_index) # 5623개

# %% 아래는 확인용 코드 (개수 등)

# 집합별 ID 개수를 출력
for index, group in enumerate(merged_sets):
    print(f"집합 {index}: {len(group)}개의 ID")
    
# %%
from collections import Counter

# 각 집합의 크기 계산
group_sizes = [len(group) for group in merged_sets]

# 집합 크기 분포 카운트
size_distribution = Counter(group_sizes)

# 출력
print("집합 크기 분포:")
for size, count in sorted(size_distribution.items()):
    print(f"{size}개의 노래를 가진 집합: {count}개")


# %%
import pandas as pd
import numpy as np

# 파일 불러오기
output_csv_file = pd.read_csv('final_lyrics_with_sets.csv')

# "Set Index" 컬럼의 데이터 확인
print(output_csv_file["Set Index"].head())  # 데이터를 출력해서 확인

# 1. Set Index 컬럼이 쉼표로 구분된 문자열이라면 이를 분리하여 리스트로 만든 후 처리
output_csv_file["Set Index"] = output_csv_file["Set Index"].astype(str)

# 각 집합 인덱스를 쉼표로 구분하여 분리한 후, 각 집합에 속한 노래의 개수를 카운트
count = output_csv_file["Set Index"].str.split(",").explode().value_counts()

# 결과 출력
print(count)

# %%
import matplotlib.pyplot as plt
# 히스토그램 그리기
plt.figure(figsize=(10, 6))  # 그래프 크기 설정
count.plot(kind='bar')  # 막대 그래프
plt.title('Distribution of Set Index')  # 제목
plt.xlabel('Set Index')  # x축 레이블
plt.ylabel('Count')  # y축 레이블
plt.xticks(rotation=45)  # x축 레이블 회전
plt.grid(axis='y')  # y축에 그리드 추가
plt.tight_layout()  # 레이아웃 조정
plt.show()  # 그래프 표시
# %%

import csv

# CSV 파일 저장 경로
output_csv_path = "one_set_tack.csv"

# CSV 파일 생성
with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    
    # 헤더 작성
    writer.writerow(["key", "id", "similarity_score"])
    
    # results 딕셔너리 순회
    for key, value in results.items():
        similarity_list = value.get("similarity_list", [])
        if similarity_list:  # similarity_list가 비어 있지 않은 경우
            for item in similarity_list:
                if len(item) == 2:  # [id, 유사도 값] 구조만 처리
                    id_value, similarity_score = item
                    writer.writerow([key, id_value, similarity_score])
        # else:  # similarity_list가 비어 있는 경우
        #     writer.writerow([key, None, None])  # None으로 빈 값 처리

print(f"CSV 파일이 생성되었습니다: {output_csv_path}")
# %%
from collections import Counter

# 결과를 담을 리스트
all_tracks = []

# Anchor와 similarity_list의 모든 id를 하나의 리스트에 모으기
anchor = [key for key in results.keys()]
all_tracks.extend(anchor)  # Anchor 추가

for key, value in results.items():
    similarity_list = value.get("similarity_list", [])
    if similarity_list:
        for item in similarity_list:
            if len(item) == 2:  # [id, similarity_score] 구조만 처리
                id_value, _ = item
                all_tracks.append(id_value)

# 등장 횟수 세기
track_counts = Counter(all_tracks)

# 등장 횟수가 1인 항목만 최종 리스트에 포함
final_track_list = [track for track, count in track_counts.items() if count == 1]

print(final_track_list)

# %%
import pandas as pd

final_file = r'C:\Users\user1\Desktop\3-2\딥러닝\project\dataset\final_final.csv'
final = pd.read_csv(final_file)
track_id = final['Track ID']

final_track_id = [track for track in track_id if track in final_track_list ]


# %%
# 저장할 텍스트 파일 경로
output_file = r'C:\Users\user1\Desktop\3-2\딥러닝\project\dataset\final_track_id.txt'

# 리스트를 텍스트 파일로 저장
with open(output_file, 'w') as file:
    for track in final_track_id:
        file.write(str(track) + '\n')  # 각 항목을 줄바꿈과 함께 저장

print(f"final_track_id 리스트가 '{output_file}'에 저장되었습니다.")

# %%
