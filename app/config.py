"""Global configuration for the pdf2sheet pipeline."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TESSDATA_DIR = str(BASE_DIR / "tessdata")
JOBS_DIR = BASE_DIR / "jobs"

# bump when extraction geometry changes: stale checkpoints are discarded
# (their crop coordinates would not match the reprocessed page images)
PIPELINE_VERSION = 4

RENDER_DPI = 300          # fallback render resolution
BATCH_SIZE = 20           # pages per batch ("n pages at a time")
MAX_WORKERS = 6           # parallel page workers (RAM-constrained host)
BLANK_INK_PX = 30         # dark pixels below this => blank cell
INK_THRESHOLD = 160       # gray value below this counts as ink
CELL_INSET = 6            # px trimmed from each cell edge (kills border lines)
STACK_GAP = 24            # white gap between stacked cells

STATE_VOCAB = ["입금", "지급"]

KB_TYPE_VOCAB = [
    "전자금융", "신규", "CD공동", "ATM이체", "ATM입금", "ATM출금", "스마트폰",
    "인터넷", "타행환", "현금입금", "현금지급", "대체입금", "대체지급", "제휴CD",
    "전화이체", "관리비", "급여", "연금", "공과금", "카드결제", "창구입금",
    "창구지급", "해지", "이자", "대출이자", "자동이체", "무통장입금", "폰뱅킹",
    "기업뱅킹", "펌뱅킹", "타행CD", "당행CD", "포인트리", "수표입금", "현금",
    "대체", "온라인", "모바일", "체크카드", "결제", "결산이자", "리브머니",
    "현금취소", "출금취소", "입금취소",
]

IBK_TYPE_VOCAB = [
    "타행환", "타행이체", "인터넷", "대체", "ATM입금", "ATM출금", "ATM이체", "타CD",
    "CD기", "펌뱅킹", "펌이체", "폰뱅킹", "텔레뱅킹", "스마트뱅킹",
    "기업스마트뱅킹", "공과금", "KFTC", "EICD", "FICD", "CC", "이체", "창구",
    "현금", "급여", "이자", "CMS", "카드", "지로", "오픈뱅킹", "스마트폰",
    "자동이체", "무통장", "대체입금", "현금입금", "IC현금", "수표", "신규",
    "해지", "통신료", "일부타점", "타행", "효성", "전화료", "나이스", "국세",
    "우체국", "자기앞", "입금취소", "API이체",
]

# ---- bank profiles ----------------------------------------------------------
# roles: datetime|date|time|state|type|amt|amt_in|amt_out|bal|branch|desc|
#        check|counter|skip
PROFILES = {
    "KB": {
        "table": "bordered",
        # column boundary ratios (x / table_width) measured at 300dpi
        "ratios": [0.000, 0.1425, 0.1868, 0.2899, 0.3965, 0.5027, 0.6174,
                   0.7774, 0.8571, 1.0000],
        "tol": 0.035,
        "header_labels": {"계좌번호": "계좌번호", "고객명": "고객명",
                          "상품명": "상품명"},
        "columns": [
            {"key": "거래일시", "role": "datetime", "lang": "eng", "wl": "0123456789/: ", "scale": 1.0, "kind": "datetime"},
            {"key": "상태",     "role": "state",    "lang": "kor_fast", "wl": None, "scale": 2.0, "kind": "vocab"},
            {"key": "거래구분", "role": "type",     "lang": "kor_fast", "wl": None, "scale": 2.0, "kind": "vocab_soft", "vocab": KB_TYPE_VOCAB},
            {"key": "거래금액", "role": "amt",      "lang": "eng", "wl": "0123456789,", "scale": 1.0, "kind": "int"},
            {"key": "표면잔액", "role": "bal",      "lang": "eng", "wl": "0123456789,", "scale": 1.0, "kind": "int"},
            {"key": "취급점",   "role": "branch",   "lang": "kor", "wl": None, "scale": 1.0, "kind": "text"},
            {"key": "적요",     "role": "desc",     "lang": "kor", "wl": None, "scale": 1.0, "kind": "text"},
            {"key": "수표번호", "role": "check",    "lang": "eng", "wl": "0123456789", "scale": 1.0, "kind": "text"},
            {"key": "이체상대계좌번호", "role": "counter", "lang": "eng", "wl": "0123456789=- ", "scale": 1.0, "kind": "text"},
        ],
    },
    "IBK": {
        "table": "headerline",   # borderless: columns located via header words
        "header_words": ["거래일", "거래시간", "출금", "입금", "잔액", "거래점",
                         "취급자", "거래코드", "거래내용", "비고", "자기앞수표",
                         "기타"],
        "min_header_hits": 8,
        "header_labels": {"계좌번호": "계좌번호", "예금주명": "고객명",
                          "계정과목": "상품명"},
        "columns": [
            {"key": "거래일",   "role": "date",    "lang": "eng", "wl": "0123456789-", "scale": 1.0, "kind": "date_dash"},
            {"key": "거래시간", "role": "time",    "lang": "eng", "wl": "0123456789:", "scale": 1.0, "kind": "time"},
            {"key": "출금",     "role": "amt_out", "lang": "eng", "wl": "0123456789,", "scale": 1.0, "kind": "int"},
            {"key": "입금",     "role": "amt_in",  "lang": "eng", "wl": "0123456789,", "scale": 1.0, "kind": "int"},
            {"key": "잔액",     "role": "bal",     "lang": "eng", "wl": "0123456789,", "scale": 1.0, "kind": "int"},
            {"key": "거래점",   "role": "branch",  "lang": "kor", "wl": None, "scale": 1.0, "kind": "text"},
            {"key": "취급자",   "role": "skip"},
            {"key": "거래코드", "role": "skip"},
            {"key": "거래내용", "role": "type",    "lang": "kor_fast", "wl": None, "scale": 2.0, "kind": "vocab_soft", "vocab": IBK_TYPE_VOCAB},
            {"key": "비고",     "role": "desc",    "lang": "kor", "wl": None, "scale": 1.0, "kind": "text"},
            {"key": "자기앞수표", "role": "check", "lang": "eng", "wl": "0123456789", "scale": 1.0, "kind": "zero_blank"},
            {"key": "기타",     "role": "skip"},
        ],
    },
}
