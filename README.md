# CS231A Final Project
Stanford CS231A 

# Experiment Result Using KITTI Depth Prediction Dataset
The evaluation metrics we use include absolute relative error (Abs Rel), square relative error (Sq Rel), root mean square error (RMSE), log root mean square error (RMSElog), and the ratio of pixels has error smaller than $1.25$, $1.25^2$, and $1.25^3$.

| Attention | Backbone | Abs Rel	| Sq Rel | RMSE	|  RMSE Log | $< 1.25$ | $< 1.25^2$ | $< 1.25 ^3$| 
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| FSRE | Resnet18 | 0.1082 | 0.7495|	4.5841 | 0.1835 | 0.8833 | 0.9637 | 0.9834 | 
| FSRE | Resnet50 | 0.1016 |	0.6998|	4.3752 | 0.1771 | $\textbf{0.8953}$ | $\textbf{0.9668}$ | 0.9842 | 
| FSRE | MobileNet v2 | 0.1117 | 0.7314 | 4.5673 | 0.1849 | 0.8747 | 0.9633 | 0.9841 | 
| FSRE | VoVnet |  0.1048 | 0.7046 | 4.4139 | 0.1781 | 0.8895 | 0.9655 | 0.9843 | 
| ROIFormer | Reset18 | 0.1071	| 0.7347 | 4.512 | 0.1816 | 0.8856 | 0.9650 | 0.9839 | 
| ROIFormer | Resnet50 |	0.1034	| 0.6863 | 4.4146 | 0.1767 | 0.8905 | 0.9660 | 0.9846 | 
| ROIFormer | Mobile Net v2 | 0.1076	| 0.7398 | 4.4912 | 0.1819 | 0.8847 | 0.9645 | 0.9839 |  
| ROIFormer Ours Attn | Resnet18 | 0.105	| 0.742	| 4.5241 | 0.1806 | 0.8883 | 0.9644 | 0.9839 | 
| ROIFormer Ours Attn | Resnet50 | 0.1034	| $\textbf{0.6765}$ | 4.4187	| 0.1775 | 0.8897 | 0.966 | $\textbf{0.9847}$ | 
| ROIFormer Ours Attn | VoVnet | $\textbf{0.1027}$	| 0.6918 | 4.3438 | 0.1769 | 0.8921 | 0.9661 | 0.9844 | 

https://docs.google.com/spreadsheets/d/1t0ran4a4eXaNEoDyErgfqtf-Tq51PjR9S1R95_9LmD8/edit?usp=sharing

# Run 
* CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 8888 train_ddp.py --data_path kitti_data/ --log_dir resnet50 --num_layers 50

# References: 
* Digging Into Self-Supervised Monocular Depth Estimation
  * https://arxiv.org/abs/1806.01260
   * https://github.com/nianticlabs/monodepth2
* FSRE Depth 
   * https://github.com/hyBlue/FSRE-Depth 
   * https://arxiv.org/abs/2108.08829
