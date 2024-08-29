from googletrans import Translator

translator = Translator(service_urls=['translate.google.com'])
trans = translator.translate('您好', src='zh-cn', dest='en')
print(trans.origin)
print(trans.text)
# export https_proxy=http://10.10.22.219:7070
# export http_proxy=http://10.10.22.219:7070
