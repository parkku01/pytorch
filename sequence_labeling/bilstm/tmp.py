f = open('../../rsc/sbd/news/news_test.txt', encoding='utf-8')
for line in f:
    line = line.strip()
    print(line+'\t'+line)