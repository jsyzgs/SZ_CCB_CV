from enum import Enum,unique

@unique
class RetMsg(Enum):
    SUCCEED = "执行成功"
    FAILED = "执行异常"

@unique
class RetCode(Enum):
    SUCCEED = "00"
    FAILED = "01"