import json
import re
def loadjsonl(filename):
    data=[]
    with open(filename, 'r') as f:
        for line in f:
            item_dict=json.loads(line)
            data.append(item_dict)
    return data
class calculator(object):
    def __init__(self,target_json_path,label_json_path):
        self.src=json.load(open(target_json_path))
        #self.label=json.load(open(label_json_path))
        self.label=[]
        #加载测试数据集
        ldict=loadjsonl(label_json_path)
        #分离出答案
        for coversation in ldict:
            ans=coversation['answer']
            #每条对话井号后面是答案
            ans=re.split('####',ans)[-1]
            ans=ans.strip()
            self.label.append(ans)
        self.ans=[]
    #获取src的答案
    def get_ans_somelikely(self):#采用一定策略获取可能的答案
        self.ans=[]
        notfound=0
        for conv in self.src:
            conv_ans=[]
            #模型的输出格式不确定
            #1.可能在输出的最后一句话进行总结，此时匹配最后两个
            #2.可能输出是在胡说八道，根本没有解决问题，如果没有匹配到数字，直接跳过
            #3.可能输出答案后自动续写，此时截断question后面的内容
            #4.可能the answer is后面是答案
            #5.可能出现算数表达式，最后一个出现的算术表达式可能是答案
            #6. #### 后面是答案
            #采用方式：
            #计算出的准确率低于但贴近真实准确率
            try:
                answer=conv['answer']
                #可能模型会自动续写，一般续写都会出现另一个Question: xxxx,将这部分截断

                temp_conv=re.split('[Qq]uestion',answer,re.IGNORECASE)
                if len(temp_conv)>1:
                    answer=temp_conv[0]

                #最后一个数字可能是答案
                #倒数第二个也可能是答案
                all_num=re.findall(r'\d+',answer)
                conv_ans.append(all_num[-1])
                if len(all_num)>1:
                    conv_ans.append(all_num[-2])

                #The answer is 后面是答案
                #可能会出现一些奇怪的符号，也需要匹配这些符号
                ans_str=re.findall(r'[tT]he answer is [^.]*\d+[^.]*.',answer)#返回所有 The answer is xxxx
                #提取出这些句子后面的数字
                if ans_str:
                    ans=[re.findall(r'\d+',i)[0] for i in ans_str]
                    conv_ans+=ans
                
                #连续的四个井号####后面是答案
                ans_str=re.findall(r'####\s+\d+',answer)
                if ans_str:#findall会返回列表，取下标0
                    ans=re.findall(r'\d+',ans_str[0])[0]
                    conv_ans.append(ans)

                #可能出现算数表达式，最后一个出现的算术表达式可能是答案
                #表达式可能出现空格，先去除掉所有空格避免干扰
                temp_conv=answer.replace(' ','')
                all_cal=re.findall(r'\d+[+\-*/]\d+=\d+',temp_conv)
                if all_cal:
                    #最后一个算数表达式
                    cal=all_cal[-1]
                    #取出等号后的值
                    ans=re.findall(r'=\d+',cal)[0][1:]
                    conv_ans.append(ans)



            except Exception as e:
                print(e)
                notfound+=1
                pass
            self.ans.append(conv_ans)
        print(f"notfound:{notfound}")
    def get_ans_alllikely(self):#获取所有可能的答案
        self.ans=[]
        for conv in self.src:
            conv_ans=[]
            #模型的输出格式不确定
            #1.可能在输出的最后一句话进行总结，此时匹配最后一个数字
            #2.可能输出是在胡说八道，根本没有解决问题，如果没有匹配到数字，直接跳过
            #采用方式：匹配所有数字，即只要出现答案就算正确（会产生误判，即中间数值与答案相同会被误判为正确）
            #计算出的准确率高于真实准确率
            try:
                conv_ans=re.findall(r'\d+',conv['answer'])
            except:
                pass
            self.ans.append(conv_ans)
    def get_ans_onelikely(self):#仅获取一个答案
        self.ans=[]
        for conv in self.src:
            conv_ans=[]
            conv_ans.append(re.findall(r'\d+',conv['answer'])[-1])
            self.ans.append(conv_ans)

    def calculate(self):
        assert len(self.ans)==len(self.label)
        acc=0
        count=0
        false_lst=[]
        for i in range(len(self.ans)):
            #if self.ans[i]==self.label[i]:
            if self.label[i] in self.ans[i]:
                acc+=1
            else:
                false_lst.append(i)
            count+=1
        print('accuracy:',acc/count)
        print(f"count:{count},false{count-acc}")
        return acc/count,false_lst

#查看效果    （在被判定为错误的答案中，显示可能为正确的答案）
def show():
    c=calculator(json_path1,json_path2)
    c.get_ans_somelikely()
    acc,false_lst1=c.calculate()
    c.get_ans_alllikely()
    acc,false_lst2=c.calculate()
    print(len(false_lst1),len(false_lst2))
    acc_likely=[i for i in false_lst1 if i not in false_lst2]
    print(f"acc_likely:{acc_likely}")
    for i in acc_likely:
        print("#################################################################")
        print(c.src[i]['answer'])
        print(c.label[i])
        print("#################################################################")

if __name__=="__main__":
    json_path1='../Result/Eight_Base_without_CoT.json'
    json_path2='../Data/GSM8K/data/test.jsonl'
    c=calculator(json_path1,json_path2)
    c.get_ans_somelikely()
    acc,false_lst1=c.calculate()
