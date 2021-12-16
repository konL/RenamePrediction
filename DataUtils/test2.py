# import jieba
# text="HelloIs_As"
#
#
# import re
#
#
# sentences=re.split('([A-Z])',text)
# sentences.append("")
# # sentences = ["".join(i) for i in zip(sentences[0::2],sentences[1::2])]
# prefix=sentences[1::2]
# sen=sentences[0::2]
# sen.pop(0)
# prefix.pop(len(prefix)-1)
# print(prefix)
# print(sen)
# sentences = ["".join(i) for i in zip(prefix,sen)]
# print(sentences)
#
#
def isMatch( s: str, p: str) -> bool:
    m, n = len(s) + 1, len(p) + 1
    dp = [[False] * n for _ in range(m)]
    dp[0][0] = True
    for j in range(2, n, 2):
        dp[0][j] = dp[0][j - 2] and p[j - 1] == '*'
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i][j - 2] or dp[i - 1][j] and (s[i - 1] == p[j - 2] or p[j - 2] == '.') \
                if p[j - 1] == '*' else \
                dp[i - 1][j - 1] and (p[j - 1] == '.' or s[i - 1] == p[j - 1])
    return dp[-1][-1]
import re
x="def funx{if(time=null){s=v;}}"
assignIndex = isMatch(x, '.* *= *.*{.*')
print(assignIndex)
index = x.find("{")
print(index)
#等号之后的{必须是第一个
isOld=True
if index != -1 and  (not assignIndex):
    # method
    x = x[index:]
    print(x)
else:
    if assignIndex:
        equidx=x.find("=")
        if x[0:equidx].find("{")!=-1:
            x=x[index:]
    else:
        if isOld:
            name = "a"
            # print(name,x)
        else:
            name = "a"
print(x)