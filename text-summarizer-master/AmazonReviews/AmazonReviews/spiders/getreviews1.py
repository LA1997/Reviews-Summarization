# -*- coding: utf-8 -*-
import scrapy
import sys
import time
import re
import pandas as pd
class Getreviews1Spider(scrapy.Spider):
    name = 'getreviews1'    
    reviews = []
    review_link = ''
    dfa = pd.read_csv('Op.csv')
    url = list(dfa['url'])[0]
    start_urls=[url]

    def __init__(self):
        allowed_domains = ['https://www.amazon.com','https://www.amazon.in']    

    def parse(self, response):
        
        self.review_link = ('/'.join(response.url.split('/')[0:3]) + response.xpath('//a[@data-hook="see-all-reviews-link-foot"]/@href').extract_first())
        time.sleep(2)
        yield scrapy.Request(self.review_link, callback=self.parse_reviews)
        
    def parse_reviews(self, response):
        nor = int(re.sub("[^\d\.]", "", response.xpath('//span[@data-hook="cr-filter-info-review-count"]//text()').extract_first().split(' ')[3]))
        pno = 3 if nor//10>3 else nor//10
        self.reviews.append(response.xpath('//span[@data-hook="review-body"]//text()').extract())
        print(self.reviews)
        for i in range(1,pno+1):
            yield scrapy.Request(self.review_link+'&pageNumber='+str(i), callback=self.parse_review_pg)
        print('\n\n\n\n\n\n\n\n\n')
        print(len(self.reviews[0]))
        print(self.reviews)
        
        rvd = {'Reviews': self.reviews[0]}
        rvdf = pd.DataFrame(rvd,columns = ['Reviews'])
        rvdf.to_csv('../ReviewsScraped.csv',index=False)
        
    def parse_review_pg(self, response):
        self.reviews.append(response.xpath('//span[@data-hook="review-body"]//text()').extract())
        yield {'Reviews':self.reviews}
    
    
        
        
