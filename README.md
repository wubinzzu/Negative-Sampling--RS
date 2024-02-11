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
   
     - SIGIR(2020) Bundle Recommendation with Graph Convolutional Networks.[[pdf]](https://arxiv.org/pdf/2005.03475.pdf) 
   
    -  KDD(2021) Curriculum Meta-Learning for Next POI Recommendation.[[pdf]](https://mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2021_Curriculum%20Meta-Learning%20for%20Next%20POI%20Recommendation.pdf)
   
     - arXiv(2022) Bayesian Negative Sampling for Recommendation[[pdf]](https://arxiv.org/pdf/2204.06520.pdf) | [[code]](https://github.com/liubin06/BNS)
     
     - CIKM(2022) A Biased Sampling Method for Imbalanced Personalized Ranking[[pdf]](https://openreview.net/pdf?id=TY1PM2Pg0Z)
    
     - RecSys(2023) Exploring False Hard Negative Sample in Cross-Domain
   Recommendation[[pdf]](https://ercdm.sdu.edu.cn/__local/3/11/1F/4813FCDEA9F8F773BE1ECBDDBA6_121A6BF4_196B08.pdf) | [[code]](https://github.com/hulkima/RealHNS)
   
     - SIGIR(2023) Neighborhood-based Hard Negative Mining for Sequential Recommendation [[pdf]](https://arxiv.org/pdf/2306.10047.pdf) | [[code]](https://github.com/floatSDSDS/GNNO)
 -  **Adversarial Sampling**
    - KDD(2018) Neural Memory Streaming Recommender Networks with Adversarial Training. [[pdf]](http://www.shichuan.org/hin/time/2018.KDD%202018%20Neural%20Memory%20Streaming%20Recommender%20Networks%20with%20Adversarial%20Training.pdf)

    -  CIKM(2018) CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.
    - IJCAI(2019) Deep Adversarial Social Recommendation.
    - CIKM(2019) Regularized Adversarial Sampling and Deep Time-aware Attention for Click-Through Rate Prediction.
    - AAAI(2019) Adversarial Binary Collaborative Filtering for Implicit Feedback.

    - TNLLS(2020) IPGAN: Generating Informative Item Pairs by Adversarial Sampling.

    - KDD(2021) PURE: Positive-Unlabeled Recommendation with Generative Adversarial Network.

    - KDD(2021) Adversarial Feature Translation for Multi-domain Recommendation.
 - **Graph-based Sampling** 
    -  WWW(2014) ACRec: a co-authorship based random walk model for academic collaboration recommendation.

   - KDD(2018) Graph Convolutional Neural Networks for Web-Scale Recommender Systems.

   - WWW(2019) SamWalker: Social Recommendation with Informative Sampling Strategy.

   - WWW(2020) Reinforced Negative Sampling over Knowledge Graph for Recommendation.

    - KDD(2021) MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems.

    - TKDE(2021) SamWalker++: recommendation with informative sampling strategy.

    - CIKM(2021) DSKReG: Differentiable Sampling on Knowledge Graph for Recommendation with Relational GNN.
 - **Additional data enhanced Sampling**
   - CIKM(2014) Leveraging Social Connections to Improve Personalized Ranking for Collaborative Filtering.

   - CIKM(2016) Social Recommendation with Strong and Weak Ties.

   - RecSys(2016) Bayesian Personalized Ranking with Multi-Channel User Feedback.

    - ICTAI(2017)Joint Geo-Spatial Preference and Pairwise Ranking for Point-of-Interest Recommendation.

   - CIKM(2017) A Personalised Ranking Framework with Multiple Sampling Criteria for Venue Recommendation.

   - WWW(2018) An Improved Sampling for Bayesian Personalized Ranking by Leveraging View Data.

   - IJCAI(2019) Reinforced Negative Sampling for Recommendation with Exposure Data.

   - IJCAI(2019) Geo-ALM: POI Recommendation by Fusing Geographical Information and Adversarial Learning Mechanism.

   - WI(2019) Bayesian Deep Learning with Trust and Distrust in Recommendation Systems.

   - arXiv(2021) Socially-Aware Self-Supervised Tri-Training for Recommendation.

   - WWW(2021) DGCN: Diversified Recommendation with Graph Convolutional Networks.

