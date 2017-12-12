import scrapy
from crawler.lib.utils import Utils
class NovelSpider(scrapy.Spider):
    name = "novel_spider"
    # utils = Utils()

    def start_requests(self):
        urls = [
            'http://pano.ktjr.com/index',
            # 'http://www.bxwx9.org/b/71/71265/20735050.html',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-1]
        filename = 'novel-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(Utils.cleanTags(response.body))
            # f.write(response.body)
        self.log('Saved file %s' % filename)