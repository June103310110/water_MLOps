# water_MLOps
code test.

#### dependency
python 3.7
```
sklearm==0.21.0
pandas==1.0.5
matplotlib==3.2.2
numpy==1.16.6
xgboost==0.90
```
## menu 
- 01-signal_noising.ipynb
  - 定義噪音，測試對訊號加噪
- 02-signal_testing.ipynb
  - 定義快速在模型上測試的函式
- 03-signal_analysis.ipynb
  - 分辨不同信號狀態，未來用以選擇不同模型或數據優化方法

### modeule
- utils.py
  - 常用函式
- noising.py
  - 加噪

## issue list:
- 投藥後到感測器發生變動之間的延遲
- 如何分析數據類型，優化數據、選擇模型
- ops system:


### 如何分析數據類型，優化數據、選擇模型
待試驗方法
1. kalman filter?
2. meta metric
3. SSIM?? (可控的評估尺標)(雖然是影像用的)
