# **Negative Sampling--RS**
This repository collects many papers related to negative sampling methods for Recommendation Systems (**RS**).
Existing negative sampling methods can be roughly divided into five categories: **Static Negative Sampling, Hard Negative Sampling, Adversarial Sampling, Graph-based Sampling and Additional data enhanced Sampling**.

## categories
 - **Static Negative Sampling**
   - UAI(2009) BPR: Bayesian Personalized Ranking from Implicit Feedback.[[pdf]](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)

   - RecSys(2012) Real-Time Top-N Recommendation in Social Streams.[[pdf]](https://www.ismll.uni-hildesheim.de/pub/pdfs/Diaz_Drumond_et_al_RECSYS2012.pdf)

   - SIGIR(2016) Fast Matrix Factorization for Online Recommendation with Implicit Feedback.[[pdf]](https://arxiv.org/pdf/1708.05024.pdf) | [[code]](https://github.com/hexiangnan/sigir16-eals)

   - RecSys(2018) Word2vec applied to Recommendation: Hyperparameters Matter.[[pdf]](https://arxiv.org/pdf/1804.04212.pdf)

    - WSDM(2021) Alleviating Cold-Start Problems in Recommendation through Pseudo-Labelling over Knowledge Graph.[[pdf]](https://arxiv.org/pdf/2011.05061.pdf)
 -  **Hard Negative Sampling**
    - SIGIR(2013) Optimizing Top-N Collaborative Filtering via Dynamic Negative Item Sampling.[[pdf]](https://wnzhang.net/papers/lambdarankcf-sigir.pdf)

    - WSDM(2014) Improving Pairwise Learning for Item Recommendation from Implicit Feedback.[[pdf]](https://www.uni-konstanz.de/mmsp/pubsys/publishedFiles/ReFr14.pdf)

    - CIKM(2015) Improving Latent Factor Models via Personalized Feature
   Projection for One Class Recommendation.[[pdf]](https://cseweb.ucsd.edu/~jmcauley/pdfs/cikm15.pdf)
   
    -  WAIM(2016) RankMBPR: Rank-aware Mutual Bayesian Personalized Ranking
   for Item Recommendation.[[pdf]](https://www.junminghuang.com/WAIM2016-yu.pdf)
   
     - AAAI(2018) WalkRanker: A Unified Pairwise Ranking Model with Multiple
   Relations for Item Recommendation.[[pdf]](https://dl.acm.org/doi/pdf/10.5555/3504035.3504352)
   
    -  arXiv(2020) Simplify and Robustify Negative Sampling for Implicit
   Collaborative Filtering.[[pdf]](https://arxiv.org/pdf/2009.03376.pdf) |  [[code]](https://github.com/dingjingtao/SRNS)
   
     - SIGIR(2020) Bundle Recommendation with Graph Convolutional Networks[[pdf]](https://arxiv.org/pdf/2005.03475.pdf) 
   
    -  KDD(2021) Curriculum Meta-Learning for Next POI Recommendation.[[pdf]](https://mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2021_Curriculum%20Meta-Learning%20for%20Next%20POI%20Recommendation.pdf)
   
     - arXiv(2022) Bayesian Negative Sampling for Recommendation.[[pdf]](https://arxiv.org/pdf/2204.06520.pdf) | [[code]](https://github.com/liubin06/BNS)
     
     - CIKM(2022) A Biased Sampling Method for Imbalanced Personalized Ranking.[[pdf]](https://openreview.net/pdf?id=TY1PM2Pg0Z)
    
     - RecSys(2023) Exploring False Hard Negative Sample in Cross-Domain
   Recommendation.[[pdf]](https://ercdm.sdu.edu.cn/__local/3/11/1F/4813FCDEA9F8F773BE1ECBDDBA6_121A6BF4_196B08.pdf) | [[code]](https://github.com/hulkima/RealHNS)
   
     - SIGIR(2023) Neighborhood-based Hard Negative Mining for Sequential Recommendation.[[pdf]](https://arxiv.org/pdf/2306.10047.pdf) | [[code]](https://github.com/floatSDSDS/GNNO)
       
     - WWW(2023) On the Theories Behind Hard Negative Sampling for Recommendation.[[pdf]](https://arxiv.org/pdf/2302.03472.pdf)
       
     - Soft BPR Loss for Dynamic Hard Negative Sampling in Recommender Systems.[[pdf]](https://arxiv.org/pdf/2211.13912.pdf)
  
     - arXir(2024) Adaptive Hardness Negative Sampling for Collaborative Filtering. [[pdf]](https://arxiv.org/pdf/2401.05191.pdf) | [[code]](https://github.com/Riwei-HEU/AHNS)
      
      - KDD(2022) FedAttack: Effective and covert poisoning attack on federated recommendation via hard sampling.[[pdf]](https://arxiv.org/pdf/2202.04975.pdf)
  
      - arXir(2024) Adaptive Hardness Negative Sampling for Collaborative Filtering. [[pdf]](https://arxiv.org/pdf/2401.05191.pdf) | [[code]](https://github.com/Riwei-HEU/AHNS)
  
      - arXir(2025)Momentum Contrastive Learning with Enhanced Negative Sampling and Hard Negative Filtering.[[pdf]](https://arxiv.org/pdf/2501.16360.pdf)
  
      - WSDM(2025)SCONE: A Novel Stochastic Sampling to Generate Contrastive Views and Hard Negative Samples for Recommendation.[[pdf]](https://dl.acm.org/doi/pdf/10.1145/3701551.3703522.pdf) | [[code]](https://github.com/jeongwhanchoi/SCONE)
    
 -  **Adversarial Sampling**
    - KDD(2018) Neural Memory Streaming Recommender Networks with Adversarial Training.[[pdf]](http://www.shichuan.org/hin/time/2018.KDD%202018%20Neural%20Memory%20Streaming%20Recommender%20Networks%20with%20Adversarial%20Training.pdf)

    - CIKM(2018) CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.[[pdf]](https://dl.acm.org/doi/10.1145/3269206.3271743)
    - IJCAI(2019) Deep Adversarial Social Recommendation.[[pdf]](https://arxiv.org/pdf/1905.13160.pdf)
    - CIKM(2019) Regularized Adversarial Sampling and Deep Time-aware Attention for Click-Through Rate Prediction.[[pdf]](https://arxiv.org/pdf/1911.00886.pdf)
    - AAAI(2019) Adversarial Binary Collaborative Filtering for Implicit Feedback.[[pdf]](https://dl.acm.org/doi/pdf/10.1609/aaai.v33i01.33015248)

    - TNLLS(2020) IPGAN: Generating Informative Item Pairs by Adversarial Sampling.[[pdf]](https://ieeexplore.ieee.org/document/9240960)

    - KDD(2021) PURE: Positive-Unlabeled Recommendation with Generative Adversarial Network.[[pdf]](https://dl.acm.org/doi/pdf/10.1145/3447548.3467234)

    - KDD(2021) Adversarial Feature Translation for Multi-domain Recommendation.[[pdf]](https://nlp.csai.tsinghua.edu.cn/~xrb/publications/KDD-2021_AFT.pdf) | [[code]](https://github.com/xiaobocser/AFT)
 - **Graph-based Sampling** 
    -  WWW(2014) ACRec: a co-authorship based random walk model for academic collaboration recommendation.[[pdf]](https://www.researchgate.net/publication/261961442_ACRec_a_co-authorship_based_random_walk_model_for_academic_collaboration_recommendation)

   - KDD(2018) Graph Convolutional Neural Networks for Web-Scale Recommender Systems.[[pdf]](https://arxiv.org/pdf/1806.01973.pdf)

   - WWW(2019) SamWalker: Social Recommendation with Informative Sampling Strategy.[[pdf]](https://jiawei-chen.github.io/paper/SamWalker.pdf) | [[code]](https://github.com/jiawei-chen/Samwalker/blob/master/Readme.md)

   - WWW(2020) Reinforced Negative Sampling over Knowledge Graph for Recommendation.[[pdf]](https://arxiv.org/pdf/2003.05753.pdf) | [[code]](https://github.com/xiangwang1223/kgpolicy)

    - KDD(2021) MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems.[[pdf]](https://static.aminer.cn/upload/pdf/1062/1446/1888/60c31feb6750f85387887e7c_0.pdf) | [[code]](https://github.com/huangtinglin/MixGCF)

    - TKDE(2021) SamWalker++: recommendation with informative sampling strategy.[[pdf]](https://arxiv.org/pdf/2011.07734.pdf) | [[code]](https://github.com/jiawei-chen/Samwalker/blob/master/Readme.md)

    - CIKM(2021) DSKReG: Differentiable Sampling on Knowledge Graph for Recommendation with Relational GNN.[[pdf]](https://arxiv.org/pdf/2108.11883.pdf)  | [[code]](https://github.com/YuWang-1024/DSKReG)
  
    - arXir(2021) Knowledge Graph-Enhanced Sampling for Conversational Recommendation System.[[pdf]](https://arxiv.org/pdf/2110.06637.pdf)
 - **Additional data enhanced Sampling**
   - CIKM(2014) Leveraging Social Connections to Improve Personalized Ranking for Collaborative Filtering.[[pdf]](https://cseweb.ucsd.edu/~jmcauley/pdfs/cikm14.pdf) 

   - CIKM(2016) Social Recommendation with Strong and Weak Ties.[[pdf]](https://mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2016_Social%20Recommendation%20with%20Strong%20and%20Weak%20Ties.pdf)

   - RecSys(2016) Bayesian Personalized Ranking with Multi-Channel User Feedback.  [[pdf]](https://dl.acm.org/doi/10.1145/2959100.2959163) | [[code]](https://github.com/mkurovski/multi_channel_bpr) 

    - ICTAI(2017)Joint Geo-Spatial Preference and Pairwise Ranking for Point-of-Interest Recommendation.[[pdf]](https://fajieyuan.github.io/papers/ICTAI16.pdf) 

   - CIKM(2017) A Personalised Ranking Framework with Multiple Sampling Criteria for Venue Recommendation.[[pdf]](https://eprints.gla.ac.uk/147491/7/147491.pdf)

   - WWW(2018) An Improved Sampling for Bayesian Personalized Ranking by Leveraging View Data.[[pdf]](http://staff.ustc.edu.cn/~hexn/papers/www18-improvedBPR.pdf)

   - IJCAI(2019) Reinforced Negative Sampling for Recommendation with Exposure Data.[[pdf]](https://www.ijcai.org/Proceedings/2019/0309.pdf) | [[code]](https://github.%20com/dingjingtao/ReinforceNS.)

   - IJCAI(2019) Geo-ALM: POI Recommendation by Fusing Geographical Information and Adversarial Learning Mechanism.[[pdf]](https://www.ijcai.org/Proceedings/2019/0250.pdf) 

   - WI(2019) Bayesian Deep Learning with Trust and Distrust in Recommendation Systems.[[pdf]](https://ieeexplore.ieee.org/document/8909635)

   - arXiv(2021) Socially-Aware Self-Supervised Tri-Training for Recommendation.[[pdf]](https://arxiv.org/pdf/2106.03569.pdf) | [[code]](https://github.com/Coder-Yu/QRec)

   - WWW(2021) DGCN: Diversified Recommendation with Graph Convolutional Networks.[[pdf]](https://arxiv.org/pdf/2108.06952.pdf) | [[code]](https://github.com/guokan987/DGCN)
- **Adaptive Negative Sampling**
   - WWW(2023) Fairly Adaptive Negative Sampling for Recommendations.[[pdf]](https://dl.acm.org/doi/pdf/10.1145/3543507.3583355)
- **Dynamic negative sampling**
  - Dynamic negative sampling for recommendation with feature matching.[[pdf]](https://link.springer.com/article/10.1007/s11042-023-17521-0)
 
  - WWW(2022) A Gain-Tuning Dynamic Negative Sampler for Recommendation.[[pdf]](https://dl.acm.org/doi/10.1145/3485447.3511956)
 
  - Towards Automated Negative Sampling in Implicit Recommendation.[[pdf]](https://arxiv.org/pdf/2311.03526.pdf)
- **Generalized Negative Sampling**
   - WWW(2020) Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations. [[pdf]](https://dl.acm.org/doi/10.1145/3366424.3386195)
 
   - arXiv(2022) Generating Negative Samples for Sequential Recommendation. [[pdf]](https://arxiv.org/pdf/2208.03645.pdf)
 
   - WI-IAT(2021) Generalized Negative Sampling for Implicit Feedback in Recommendation.[[pdf]](https://dl.acm.org/doi/10.1145/3486622.3493998)
 
   - CIKM(2023) Batch-Mix Negative Sampling for Learning Recommendation Retrievers. [[pdf]](https://dl.acm.org/doi/abs/10.1145/3583780.3614789)
 
   - Reinforced PU-learning with Hybrid Negative Sampling Strategies for Recommendation.[[pdf]](https://dl.acm.org/doi/abs/10.1145/3582562)
 
   - ISAIEE(2022) Time and Space Aggregation Recommendation Model Based on Synthetic Negative Samples.[[pdf]](https://ieeexplore.ieee.org/document/10071152)
 
   - ICDM(2022) MixDec Sampling: A Soft Link-based Sampling Method of Graph Neural Network for Recommendation.[[pdf]](https://ieeexplore.ieee.org/document/10027691)
 
   - CSSE(2022) Hybrid Sampling Light Graph Collaborative Filtering for Social Recommendation.[[pdf]](https://dl.acm.org/doi/10.1145/3569966.3570002)

- **未分类**
   - CIKM(2016) Tag-Aware Personalized Recommendation Using a Deep-Semantic Similarity Model with Negative Sampling.[[pdf]](https://dl.acm.org/doi/10.1145/2983323.2983874)
   - 专家系统(2022) GANRec: A negative sampling model with generative adversarial network for recommendation.[[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S095741742202173X) | [[code]](https://github.com/Yangzhi22/GANRec)
     
   - Effective and efficient negative sampling in metric learning based recommendation.[[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S0020025522004650) 

   -  RecSys(2023) gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling.[[pdf]](https://arxiv.org/pdf/2308.07192.pdf)

   - Collaborative knowledge-aware recommendation based on neighborhood negative sampling.[[pdf]](https://dl.acm.org/doi/abs/10.1016/j.is.2023.102207)

    - SIGIR(2022) Rule-Guided Knowledge-Graph based Negative Sampling for Outfit Recommendation.[[pdf]](https://sigir-ecom.github.io/ecom22Papers/paper_8739.pdf)

   - A Negative Sampling-Based Service Recommendation Method.[[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-24383-7_1)

   - RecSys(2021) A Case Study on Sampling Strategies for Evaluating Neural Sequential Item Recommendation Models.[[pdf]](https://arxiv.org/pdf/2107.13045.pdf)

   - SIAM(2023) UFNRec: Utilizing False Negative Samples for Sequential Recommendation.[[pdf]](https://arxiv.org/pdf/2208.04116.pdf) | [[code]](https://github.com/UFNRec-code/UFNRec)

   - Development of an offline OOH advertising recommendation system using negative sampling and deep interest network 2024 [[pdf]](https://dl.acm.org/doi/10.1007/s11042-023-16083-5)

   - TCSS(2024) Graph Contrastive Learning With Negative Propagation for Recommendation.[[pdf]](https://ieeexplore.ieee.org/document/10419035)

    - WAIM(2022) User Multi-behavior Enhanced POI Recommendation with Efficient and Informative Negative Sampling.[[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-25201-3_11)

     - Mutual Fund Recommendation Based on Robust Negative Sampling with Graph-Cut Algorithm  [[pdf]](https://kdf-workshop.github.io/kdf23/assets/images/kdf_2.pdf)

   - SIGIR (2023) Exploring the Impact of Negative Sampling on Patent Citation Recommendation.[[pdf]](https://ceur-ws.org/Vol-3604/paper3.pdf)

   - Reinforcement Learning-Based Explainable Recommendation over Knowledge Graphs with Negative Sampling.[[pdf]](https://ieeexplore.ieee.org/document/10189524)

    -  Negative can be positive: Signed graph neural networks for recommendation.[[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S0306457323001401)

    - MSN(2019) Addressing the Conflict of Negative Feedback and Sampling for Online Ad Recommendation in Mobile Social Networks.[[pdf]](https://ieeexplore.ieee.org/document/9066164)

    -  Gan-based recommendation with positive-unlabeled sampling.[[pdf]](https://arxiv.org/pdf/2012.06901.pdf)
 
   - WSDM(2023) Relation Preference Oriented High-order Sampling for Recommendation.[[pdf]](https://dl.acm.org/doi/10.1145/3539597.3570424) | [[code]](https://github.com/RPHS/RPHS.git)
 
   - RecSys(2023) Augmented Negative Sampling for Collaborative Filtering.[[pdf]](https://dl.acm.org/doi/10.1145/3604915.3608811) | [[code]](https://github.com/Asa9aoTK/ANS-Recbole.)
   
   - WSDM(2023) Disentangled Negative Sampling for Collaborative Filtering. [[pdf]](https://dl.acm.org/doi/abs/10.1145/3539597.3570419) | [[code]](https://github.com/Riwei-HEU/DENS)
 
   - CIKM(2024)Improved Estimation of Ranks for Learning Item Recommenders with Negative Sampling. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3627673.3679943)
  
   - ACM(2024) VAE*: A Novel Variational Autoencoder via Revisiting Positive and Negative Samples for Top-N Recommendation.[[pdf]](https://dl.acm.org/doi/10.1145/3680552)
 
   - KDD(2023)Robust Positive-Unlabeled Learning via Noise Negative Sample Self-correction.[[pdf]](https://dl.acm.org/doi/pdf/10.1145/3580305.3599491.pdf) | [[code]](https://github.com/woriazzc/Robust-PU.)
 
   - arXiv(2023)Reducing Popularity Bias in Recommender Systems through AUC-Optimal Negative Sampling.[[pdf]](https://arxiv.org/pdf/2306.01348)
   - RECSYS(2023)Scaling Session-Based Transformer Recommendations using Optimized Negative Sampling and Loss Functions.[[pdf]](https://arxiv.org/pdf/2307.14906)
   - WWW(2024)Diffusion-based Negative Sampling on Graphs for Link Prediction.[[pdf]](https://dl.acm.org/doi/pdf/10.1145/3589334.3645650)
   - RECSYS(2023)gSASRec: reducing overconfidence in sequential recommendation trained with negative sampling.[[pdf]](https://dl.acm.org/doi/10.24963/ijcai.2024/939)
   - IEEE(2025)Exploration and Exploitation of Hard Negative Samples for Cross-Domain Sequential Recommendation.[[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10879516)
