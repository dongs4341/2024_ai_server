## Commit Rule

### Commit Message Structure

> type: Subject
> 
> 
> body
> 
> footer
> 

### Commit Subject  Rule

> 커밋 제목은 최대 50자 
제목 끝에 마침표(.) 금지
Commit 메세지
> 

### Commit Type Rule

✂️[chore]: 코드 수정, 내부 파일 수정

✨[feat]: 새로운 기능 구현

🎨[style]: 스타일 관련 기능.(코드의 구조/형태 개선)

➕[add]: Feat 이외의 부수적인 코드 추가, 라이브러리 추가

🔧[file]: 새로운 파일 생성, 삭제 시

🐛[fix]: 버그, 오류 해결

🔥[del]: 쓸모없는 코드/파일 삭제

📝[docs]: README나 WIKI 등의 문서 개정

💄[mod]: storyboard 파일,UI 수정한 경우

✏️[correct]: 주로 문법의 오류나 타입의 변경, 이름 변경 등에 사용

🚚[move]: 프로젝트 내 파일이나 코드(리소스)의 이동

⏪️[rename]: 파일 이름 변경이 있을 때 사용

⚡️[improve]: 향상이 있을 때 사용

♻️[refactor]: 전면 수정이 있을 때 사용

🔀[merge]: 다른브렌치를 merge 할 때 사용

✅ [test]: 테스트 코드를 작성할 때 사용


### **Commit Body Rule**

> 제목과 본문을 빈 행으로 분리<br>
입력본문은 한 줄 최대 72자 입력<br>
본문을 사용하여 변경 한 내용과 이유 설명(어떻게 보다는 무엇과 왜를 설명)<br>
한글로 작성<br>


### 필요한 패키지 설치
pip install -r requirements.txt #패키지 설치

### FastAPI 실행
uvicorn app.main:app --reload

