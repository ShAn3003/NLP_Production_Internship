import random 
def generate_prompt(question = "",CoT=True, Shot=8 ,Instruction = True , fixed = True , train_data = None):
    example_without_cot = [{'Question': 'Jerry’s two daughters play softball on different teams. They each have 8 games this season. Each team practices 4 hours for every game they play. If each game lasts for 2 hours, how many hours will Jerry spend at the field watching his daughters play and practice altogether?',
    'Answer': 'Jerry will spend 8 games x 2 hours per game = <<8*2=16>>16 hours watching one daughter play her games.\nHe will spend 16 x 2 = <<16*2=32>>32 hours watching both daughters play their games.\nHe will spend 8 games x 4 hours of practice = <<8*4=32>>32 hours watching one daughter practice.\nHe will spend 32 x 2 = <<32*2=64>>64 hours watching both daughters practice.\nHe will spend a total of 32 hours watching games + 64 hours watching practice = <<32+64=96>>96 hours.\n#### 96'},
    {'Question': "A class has 60 students. The number of students who bring their lunch is thrice the number of those who eat in the school cafeteria. The rest of the students don't eat lunch. If 10 students eat in the school cafeteria, how many don't eat lunch?",
    'Answer': "Multiplying the number of students eating at the cafeteria by three means that 3 * 10 = <<3*10=30>>30 students bring lunch.\nThe total number of students that eat lunch is 10 + 30 = <<10+30=40>>40 students\nSince the class has 60 students, those who don't eat lunch are 60-40 = <<60-40=20>>20 students.\n#### 20"},
    {'Question': 'Lily has 5 lottery tickets to sell.  She sells the first ticket for $1.  She then sells each successive ticket for a dollar more than the previous ticket. She plans to keep a $4 profit and give the remaining money as the prize. How much money will the winner of the lottery receive?',
    'Answer': 'The second ticket sold will cost $1 + $1 = $<<1+1=2>>2.\nThe third ticket sold will cost $2 + $1 = $<<2+1=3>>3.\nThe fourth ticket sold will cost $3 + $1 = $<<3+1=4>>4.\nThe fifth ticket sold will cost $4 + $1 = $<<4+1=5>>5.\nThe total money collected is $1 + $2 + $3 + $4 + $5 = $<<1+2+3+4+5=15>>15.\nAfter taking profit, the total prize money will be $15 - $4 = $<<15-4=11>>11.\n#### 11'},
    {'Question': "Smaug the dragon hoards 100 gold coins, 60 silver coins, and 33 copper coins. If each silver coin is worth 8 copper coins, and each gold coin is worth 3 silver coins, what is the total value of Smaug's hoard expressed as a number of copper coins?",
    'Answer': 'First figure out how many silver coins the 100 gold coins are worth by multiplying the number of gold coins by the exchange rate between gold and silver: 100 gold * 3 silver/gold = <<100*3=300>>300 silver\nThen add the value of the gold in silver to the number of silver coins to find the total value of the gold and silver expressed in silver coins: 300 silver + 60 silver = <<300+60=360>>360 silver\nNow multiply that value by the exchange rate between silver and copper to express its value in terms of copper coins: 360 silver * 8 copper/silver = <<360*8=2880>>2880 copper\nThen add the value of the gold and silver expressed in copper coins (the value from the last step) to the number of copper coins to find the total value of the hoard: 2880 + 33 = <<2880+33=2913>>2913\n#### 2913'},
    {'Question': "Peggy is moving and is looking to get rid of her record collection. Sammy says that he will buy all of them for 4 dollars each. Bryan is only interested in half of the records but will offer 6 dollars each for the half that he is interested in and 1 dollar each for the remaining half that he is not interested in with the hopes that he can resell them in bulk later. If Peggy has 200 records, what is the difference in profit between Sammy versus Bryan's deal?",
    'Answer': "Sammy is offering to take the whole collection of 200 records and pay Peggy 4 dollars each for them which would net Peggy 200 * 4=<<200*4=800>>800 dollars for her entire record collection.\nBryan is willing to buy Peggy's entire record collection but at two different price points, half at one point and half at another. Half of Peggy's record collection is 200/2=<<200/2=100>>100, which means that 100 records will sell for one price and 100 records will sell for another price.\nBryan is willing to pay more for the half of the record collection that he is interested in so Peggy would net 100 * 6=<<100*6=600>>600 dollars for the first half of her record collection.\nFor the half of the collection that Bryan is just planning on reselling at a later date, he is willing to offer Peggy 100 *1=<<100*1=100>>100 dollars to take off of her hands.\nIn total Bryan is willing to offer Peggy 600+100=<<600+100=700>>700 dollars for her entire record collection.\nIf Sammy is offering 800 dollars to buy Peggy's entire record collection and Bryan is offering 700 dollars for Peggy's entire record collection, then Peggy's net profit would be 800-700=<<800-700=100>>100 dollars more by taking Sammy's deal instead of Bryan's deal.\n#### 100"},
    {'Question': 'Mary bought six apples from the store. From the apples she bought, for each that Mary ate, she planted two trees from the remaining ones. How many apples did Mary eat?',
    'Answer': 'She planted eight trees. This means she used half of that amount of apples, which is 8 trees / 2 trees/apple = <<8/2=4>>4 apples.\nThat means that she planted four of the six apples she bought, leaving only 6 apples - 4 apples = <<6-4=2>>2 apples to be eaten.\n#### 2'},
    {'Question': 'Mary Anne drinks 1/5 of a bottle of sparkling water every night at dinner.  If each bottle costs her $2.00, how much does she spend on sparkling water every year?',
    'Answer': 'She drinks 1/5 of a bottle of sparkling water every night so over an entire year she drinks .2*365 = <<1/5*365=73>>73 bottles of sparkling water\nEvery bottle costs $2.00 and she drinks 73 bottles a year so she spends 2*73 = $<<2*73=146.00>>146.00 a year on sparkling water\n#### 146'},
    {'Question': "Jackson is making dinner. He makes a salad out of lettuce (50 calories), carrots (twice the calories of the lettuce) and dressing (210 calories). He also makes a pizza with 600 calories for the crust, 1/3 the crust's calories for the pepperoni, and 400 calories for the cheese. If Jackson eats 1/4 of the salad and 1/5 of the pizza, how many calories does he eat?",
    'Answer': "First find the number of calories in the carrots: 50 calories * 2 = <<50*2=100>>100 calories\nThen find the total calories in the salad: 100 calories + 50 calories + 210 calories = <<100+50+210=360>>360 calories\nThen find the number of calories in the pepperoni: 1/3 * 600 calories = <<1/3*600=200>>200 calories\nNow find the total number of calories in the pizza: 200 calories + 600 calories + 400 calories = <<200+600+400=1200>>1200 calories\nNow find how many calories are in Jackson's portion of the salad: 360 calories * 1/4 = <<360*1/4=90>>90 calories\nNow find how many calories are in Jackson's portion of the pizza: 1200 calories * 1/5 = <<1200*1/5=240>>240 calories\nNow find the total calories Jackson ate: 90 calories + 240 calories = <<90+240=330>>330 calories\n#### 330"}]

    example_special_cot = [{'Question': 'There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?',
    'Answer': 'There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.'},
    {'Question': 'If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?',
    'Answer': 'There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.'},
    {'Question': 'Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?',
    'Answer': 'Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.'},
    {'Question': 'Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?',
    'Answer': 'Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.'},
    {'Question': 'Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?',
    'Answer': 'Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.'},
    {'Question': 'There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?',
    'Answer': 'There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.'},
    {'Question': 'Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?',
    'Answer': 'Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.'},
    {'Question': 'Olivia has $23. She bought five bagels for $3 each. How much money does she have left?',
    'Answer': 'Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.'}]

    Instruction_prefix = """Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n"""
    if fixed:
        if Instruction and ( not CoT) :
            exampler_shot = ""
            for i in range(Shot):
                exampler_shot+=f"### Instruction:\n{example_without_cot[i]['Question']}\n\n### Response:{example_without_cot[i]['Answer']}\n\n"
            prompt = Instruction_prefix+exampler_shot+f"### Instruction:\n{question}\n\n### Response:"
            return prompt
        elif Instruction and CoT :
            exampler_shot = ""
            for i in range(Shot):
                exampler_shot+=f"### Instruction:\n{example_special_cot[i]['Question']}\n\n### Response: Let's think step by step.\n{example_special_cot[i]['Answer']}\n\n"
            prompt = Instruction_prefix+exampler_shot+f"### Instruction:\n{question}\n\n### Response: Let's think step by step.\n"
            return prompt
        elif (not Instruction) and ( not CoT):
            exampler_shot = ""
            for i in range(Shot):
                exampler_shot+=f"Question:\n{example_without_cot[i]['Question']}\n\nAnswer:{example_without_cot[i]['Answer']}\n\n"
            prompt = exampler_shot+f"Question:\n{question}\n\nAnswer:"
            return prompt
        elif (not Instruction) and  CoT:
            exampler_shot = ""
            for i in range(Shot):
                exampler_shot+=f"Question:\n{example_special_cot[i]['Question']}\n\nAnswer:{example_special_cot[i]['Answer']}\n\n"
            prompt = exampler_shot+f"Question:\n{question}\n\nAnswer:"
            return prompt
    else:
        if train_data is None:
            raise ValueError("train_data must be provided if fixed=False")
        train_data_size = len(train_data)
        ran_index = random.sample(range(train_data_size), Shot)
        if Instruction and ( not CoT) :
            exampler_shot = ""
            for i in ran_index:
                exampler_shot+=f"### Instruction:\n{train_data[i]['question']}\n\n### Response:{train_data[i]['answer']}\n\n"
            prompt = Instruction_prefix+exampler_shot+f"### Instruction:\n{question}\n\n### Response:"
            return prompt
        elif Instruction and CoT :
            exampler_shot = ""
            for i in range(Shot):
                exampler_shot+=f"### Instruction:\n{example_special_cot[i]['Question']}\n\n### Response: Let's think step by step.\n{example_special_cot[i]['Answer']}\n\n"
            prompt = Instruction_prefix+exampler_shot+f"### Instruction:\n{question}\n\n### Response: Let's think step by step.\n"
            return prompt
        elif (not Instruction) and ( not CoT):
            exampler_shot = ""
            for i in ran_index:
                exampler_shot+=f"Question:\n{train_data[i]['question']}\n\nAnswer:{train_data[i]['answer']}\n\n"
            prompt = exampler_shot+f"Question:\n{question}\n\nAnswer:"
            return prompt
        elif (not Instruction) and  CoT:
            exampler_shot = ""
            for i in range(Shot):
                exampler_shot+=f"Question:\n{example_special_cot[i]['Question']}\n\nAnswer:{example_special_cot[i]['Answer']}\n\n"
            prompt = exampler_shot+f"Question:\n{question}\n\nAnswer:"
            return prompt 