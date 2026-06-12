# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 오용석
- 리뷰어 : 유비


# PRT(Peer Review Template)
- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 잘 첨부되어 있고 결과 그래프로 학습루프의 결과를 보여주고 있
     <img width="533" height="704" alt="스크린샷 2026-06-05 103051" src="https://github.com/user-attachments/assets/16846590-d03d-4986-8e43-e139200cc169" />
     <img width="482" height="806" alt="스크린샷 2026-06-05 103120" src="https://github.com/user-attachments/assets/e0db7981-70c4-4579-b74f-fc40bfcecdf8" />
     <img width="480" height="839" alt="스크린샷 2026-06-05 103144" src="https://github.com/user-attachments/assets/c8f3ff07-420c-4db2-bd69-054fe47bfb93" />
     <img width="425" height="587" alt="스크린샷 2026-06-05 101937" src="https://github.com/user-attachments/assets/fc8ad7bd-af6f-40e1-a96a-33a495b55da0" />

- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - Resnet의 본질인 skip connection, 차원을 맞추는 부분이 모두 이 블록에 있고, 전체 모델이 이 블록의 반복으로 만들어지기 때문에 핵심적임
    - 해당 코드 블럭에 doc string을 작성해두었음
    - 첨부한 블록 이외에 전체적으로 주석과 각 블럭 속 코드의 기능, 존재 이유, 작동 원리 등을 친절하게 기술하였음
     <img width="937" height="382" alt="스크린샷 2026-06-05 101650" src="https://github.com/user-attachments/assets/2075361b-4019-4bc5-8440-c6d700754a35" />

- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 주석으로 학습 환경에 따라 조정한 부분의 변화된 점을 기록 남김(새로운 시도를 함)
     <img width="422" height="46" alt="스크린샷 2026-06-05 105926" src="https://github.com/user-attachments/assets/b8d38d57-36ca-44f5-ba73-e5b42c5f9cf4" />
     <img width="461" height="53" alt="스크린샷 2026-06-05 105945" src="https://github.com/user-attachments/assets/e15b24c5-4977-406e-95d3-e045ec3fa2bb" />
        
- [x]  **4. 회고를 잘 작성했나요?**
    - 회고는 입력되어 있지는 않으나 loss, accuracy를 시각화 하는 그래프를 만들었고 위에 언급한 것처럼 블록에 대한 설명, 주석이 매우 자세하게 작성되어 있어 보기에 용이함
     <img width="937" height="382" alt="스크린샷 2026-06-05 101650" src="https://github.com/user-attachments/assets/11db02d1-1fb7-4871-be08-a2f491c3eb4f" />
     <img width="425" height="587" alt="스크린샷 2026-06-05 101937" src="https://github.com/user-attachments/assets/ed1df446-e772-4f3e-897f-5068fc3e0455" />
   
- [x]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 잘 준수하였음
    - _make_layer함수로 반복 블록을 깔끔하게 모듈화하였고 expansion 변수로 BasicBlock(1)/Bottleneck(4) 구분한 점도 범용적임
     <img width="397" height="106" alt="스크린샷 2026-06-05 100902" src="https://github.com/user-attachments/assets/8a889fa3-9c0d-437d-82d3-8eff34450d92" />
     <img width="950" height="845" alt="스크린샷 2026-06-05 101135" src="https://github.com/user-attachments/assets/8ae4ac03-2565-40c8-9591-318dc74e595c" />



# 회고(참고 링크 및 코드 개선)
```
# 코드가 전반적으로 깔끔하고 반복되는 부분을 줄이는 코드 블록을 구성한걸 볼 수 있었다.
# 주석을 적재적소에 맞게 작성하여 비교적 쉽고 빠른 이해를 도왔다.
```
