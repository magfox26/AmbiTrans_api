# AmbiTrans_api  

## 目录  
api存放调用api的代码:  
[translate.py](https://github.com/magfox26/AmbiTrans_api/blob/main/api/translate.py)包括的模型：*gpt-4o-2024-11-20*，*o1-2024-12-17*，*qwen-vl-max*，*qvq-max*，*gemini-2.0-flash-001*，*gemini-2.5-flash-preview-04-17*，*gemini-2.5-pro-preview-05-06*，*anthropic.claude-3-7-sonnet-20250219-v1:0*  

## 可用参数   
--model 可以指定模型名称，如果指定多个模型中间用空格隔开，指定全部用all  
choices=['gpt-4o', 'o1', 'qvq', 'qwen','gemini-2.0-flash', 'claude-3-7-sonnet','gemini-2.5-flash', 'gemini-2.5-pro','all']  

## 日志   
### 2025年5月27日   
第一个窗口运行：  
`python translate.py --model gpt-4o o1 qvq`

第二个窗口运行：  
`python translate.py --model qwen gemini-2.0-flash gemini-2.5-flash`

第三个窗口运行：  
`python translate.py --model gemini-2.5-pro claude-3-7-sonnet`
