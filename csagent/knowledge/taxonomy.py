"""메타데이터 분류 기준 — 사람이 검토·수정하는 매핑표는 전부 이 파일에 둔다.

필드와 쓰임
- audience  (소비자 consumer / 가맹점 merchant / 셀러 seller): 게시 워크스페이스로 정해지는 사실값 → 검색 필터
- companies (토스 / 토스뱅크 / 토스페이 / 토스플레이스 / 토스모바일): 게시 워크스페이스 → 검색 필터
- domains   (업무 분야, 아래 목록): 카테고리 매핑·키워드로 **추정**한 값 → 검색에는 쓰지 않는다.
            평가 결과를 분야별로 나눠 볼 때만 쓰는 분석용 라벨.
- service   (처리 주체: 토스 / 토스뱅크 / 토스증권 / 토스쇼핑 …): 제목·게시 탭·본문 번호로 **추정**한 값.
            게시 탭 ≠ 처리 주체라서 따로 둔다 (쇼핑 FAQ는 「토스」 탭, 토스뱅크 카드 FAQ는 두 탭 모두). 근거가 없으면 "".
- contacts  (본문에 나온 토스 계열 고객센터 번호): 아래 TOSS_CONTACTS에 있는 번호만. 카드사·은행·신용평가사 번호는 넣지 않는다.
"""

from __future__ import annotations

import re

CONSUMER_DOMAINS = ["송금·계좌", "카드", "대출", "투자", "결제·페이", "쇼핑", "계정·보안", "혜택·서비스", "통신"]
MERCHANT_DOMAINS = ["가맹점"]
SELLER_DOMAINS = ["입점·상점", "상품", "주문·배송·CS", "정산", "페널티", "광고"]
OTHER = "기타"

# ---------------------------------------------------------------- FAQ

# FAQ 카테고리 ID → 도메인. 이름 없는 ID라서 각 카테고리의 조회수 상위 질문을 보고 붙였다.
# None = 여러 분야가 섞인 범용 카테고리(인기 질문 모음 등) → 제목 키워드 규칙으로 판단.
FAQ_CATEGORY_DOMAINS: dict[int, str | None] = {
    # 범용 (분야 혼합)
    12: None, 8: None, 185: None, 237: None,
    # 송금·계좌: 송금, 통장·증명서, 한도제한계좌, 외화·해외송금, 해외납세의무
    5: "송금·계좌", 6: "송금·계좌", 23: "송금·계좌", 21: "송금·계좌", 273: "송금·계좌", 343: "송금·계좌",
    # 카드: 체크카드, 유스·아이카드, 제휴카드
    7: "카드", 27: "카드", 302: "카드", 239: "카드",
    # 대출: 신용·전월세·주담대·대환
    25: "대출", 29: "대출", 335: "대출", 233: "대출",
    # 투자: 토스증권 전반 (해외주식, 배당, 조건주문, 세금, WTS, 담보대출, 환전)
    119: "투자", 103: "투자", 105: "투자", 10: "투자", 117: "투자", 396: "투자", 101: "투자",
    304: "투자", 107: "투자", 113: "투자", 351: "투자", 115: "투자", 99: "투자", 339: "투자", 367: "투자",
    # 쇼핑
    298: "쇼핑",
    # 계정·보안: 페이스패스, 안심보상·사기 피해, 알림·고객확인
    341: "계정·보안", 13: "계정·보안", 291: "계정·보안", 14: "계정·보안",
    # 혜택·서비스: 환급 찾기, 이벤트, 지원금, 프라임·포인트
    365: "혜택·서비스", 345: "혜택·서비스", 392: "혜택·서비스", 163: "혜택·서비스",
    # 통신: 토스모바일
    165: "통신",
    # 가맹점: 토스플레이스(단말기·포스·키오스크), 토스페이 가맹(계약·정산·부가세·전자계약)
    155: "가맹점", 167: "가맹점", 169: "가맹점", 171: "가맹점", 173: "가맹점", 175: "가맹점",
    177: "가맹점", 147: "가맹점", 135: "가맹점", 263: "가맹점", 129: "가맹점", 294: "가맹점",
    265: "가맹점", 137: "가맹점", 131: "가맹점",
}

# 카테고리로 정해지지 않을 때 제목 키워드로 판단 (위에서부터 먼저 맞는 것 하나)
FAQ_KEYWORD_DOMAINS: list[tuple[str, str]] = [
    (r"쇼핑|배송|반품|교환|브랜드콘", "쇼핑"),
    (r"카드", "카드"),
    (r"대출|금리|상환|신용점수|신용정보|KCB|NICE", "대출"),
    (r"증권|주식|배당|ETF|ETN", "투자"),
    (r"요금제|개통|eSIM|유심", "통신"),
    (r"송금|계좌|이체|통장|ATM|출금|입금|외화|환전|한도|용돈|거래내역", "송금·계좌"),
    (r"결제|페이|영수증", "결제·페이"),
    (r"가입|로그인|인증|명의|사기|비밀번호|탈퇴|해킹|차단|개명|고객확인", "계정·보안"),
    (r"환급|포인트|이벤트|프라임|만보기|걸음|티머니|혜택|복권|퀴즈", "혜택·서비스"),
]

# 토스페이·토스플레이스 워크스페이스 FAQ는 가맹점 대상 — 단, 결제한 사람이 묻는 질문만 소비자로 본다
# (예: 「토스페이로 금액이 출금되었어요」 「토스페이로 결제한 현금영수증 발행」)
_MERCHANT_WORKSPACES = {"토스페이", "토스플레이스"}
_CONSUMER_IN_MERCHANT = re.compile(r"토스페이로 (금액이 출금|결제한)")


def faq_audience(companies: list[str], title: str) -> str:
    if set(companies) & _MERCHANT_WORKSPACES and not _CONSUMER_IN_MERCHANT.search(title):
        return "merchant"
    return "consumer"


def faq_domains(category_ids: list[int], title: str, audience: str) -> list[str]:
    if audience == "merchant":
        return ["가맹점"]
    mapped = {FAQ_CATEGORY_DOMAINS.get(c) for c in category_ids} - {None, "가맹점"}
    if mapped:
        return sorted(mapped)
    for pattern, domain in FAQ_KEYWORD_DOMAINS:
        if re.search(pattern, title):
            return [domain]
    return [OTHER]


# ---------------------------------------------------------------- 판매자 문서

# 문서 경로(doc_id의 "shop:" 뒤) 앞부분 → 도메인. 더 구체적인 규칙을 위에 둔다.
SHOP_PATH_DOMAINS: list[tuple[str, str]] = [
    ("onboarding__settlement", "정산"),
    ("shopping-operations__order__settlement", "정산"),
    ("ads__", "광고"),
    ("penalty__", "페널티"),
    ("onboarding__", "입점·상점"),
    ("support__faq__onboarding", "입점·상점"),
    ("support__solution", "입점·상점"),
    ("shopping-operations__shop__my-store", "입점·상점"),
    ("products__", "상품"),
    ("support__faq__products", "상품"),
    ("shopping-operations__catalog__", "상품"),
    ("shopping-operations__price__", "상품"),
    ("shopping-operations__products__", "상품"),
    ("shopping-operations__order__", "주문·배송·CS"),
    ("shopping-operations__fulfillment__", "주문·배송·CS"),
    ("shopping-operations__shop__support", "주문·배송·CS"),
    ("shopping-operations__support__faq-1", "주문·배송·CS"),
]


def shop_domains(doc_path: str) -> list[str]:
    for prefix, domain in SHOP_PATH_DOMAINS:
        if doc_path.startswith(prefix):
            return [domain]
    return [OTHER]


# ---------------------------------------------------------------- 처리 주체·연락처

# 토스 계열 고객센터 번호 → 처리 주체. 본문 문장으로 주인을 확인한 번호만 (근거 FAQ).
# 본문의 다른 번호(카드사·은행·KCB/NICE 등)는 그 기관 번호라 여기 넣지 않는다.
# contacts는 「이 문서가 안내하는 번호」라 처리 주체와 다를 수 있다 (faq:6705 「토스뱅크 통장이 지급정지 되었다면 1661-7654」).
TOSS_CONTACTS: dict[str, str] = {
    "1599-4905": "토스",          # faq:157 「토스 고객센터(1599-4905)」
    "1661-4905": "토스",          # faq:437 「토스 고객보호센터(1661-4905)」 (본인 확인 전화)
    "1661-7654": "토스뱅크",      # faq:1253 「토스뱅크 고객센터(1661-7654)」
    "1599-7987": "토스증권",      # faq:207 「토스증권 … 전화 : 1599-7987」
    "1644-2944": "토스쇼핑",      # faq:4705 「토스쇼핑 고객센터(1644-2944)」
    "1660-1114": "토스모바일",    # faq:4541 「토스모바일 고객센터 1660-1114」
    "1644-2955": "토스플레이스",  # faq:4461 토스플레이스 탭 「전화상담 : 1644-2955」
}

# 제목에 이름이 나오면 그 회사가 처리 주체 (「토스쇼핑 배송지를 어떻게 바꾸나요?」)
_TITLE_SERVICES = ["토스쇼핑", "토스뱅크", "토스증권", "토스모바일"]
# 그 탭에만 게시된 FAQ는 탭이 곧 주체
_SINGLE_TAB_SERVICES = {"토스뱅크", "토스페이", "토스플레이스", "토스모바일"}


def contacts(body: str) -> list[dict]:
    return [{"service": s, "phone": p} for p, s in TOSS_CONTACTS.items() if p in body]


def faq_service(companies: list[str], title: str, audience: str, domains: list[str], body: str) -> str:
    named = [s for s in _TITLE_SERVICES if s in title]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:  # 「토스에 개인 추심 요청은? (토스뱅크 및 토스증권은 별도 요청 필요)」
        return "토스" if "토스" in companies else ""
    tabs = set(companies)
    if audience == "consumer" and tabs & _MERCHANT_WORKSPACES:
        return "토스"  # 결제한 사람의 문의 (faq:5884 「토스 고객센터 1599-4905」로 안내)
    if len(tabs) == 1 and tabs <= _SINGLE_TAB_SERVICES:
        return companies[0]
    if "토스모바일" in tabs:
        return "토스모바일"
    if "투자" in domains:  # 투자 = 토스증권 카테고리 (FAQ_CATEGORY_DOMAINS)
        return "토스증권"
    if tabs == {"토스", "토스뱅크"}:
        # 두 탭에 다 게시 — 본문 안내 번호가 한 곳뿐일 때만 그쪽으로, 아니면 판단 보류
        found = {c["service"] for c in contacts(body)}
        return found.pop() if len(found) == 1 else ""
    return "토스" if tabs == {"토스"} else ""
