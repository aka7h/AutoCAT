cat_opt <- function(tr_d, te_d, 
                    tr_label,
                    evalmetric,
                    lossfunction=NULL,
                    cat_features=NULL,
                    depth_range = c(4L, 6L),
                    eta_range = c(1e-3, 1L),
                    iter_range = c(1000L, 5000L),
                    l2_reg_range = c(1e-3,1e-1),
                    rsm_range = c(0.7,1L),
                    border_count = 255,
                    init_points = 4,
                    n_iter = 10,
                    acq = "ei",
                    kappa = 2.576,
                    eps = 0.0,
                    optkernel = list(type = "exponential", power = 2),
                    grid_len = NULL
){
  y_label <- rlang::enquo(tr_label)
  tr_y <- (tr_d %>% select(!! y_label))[[1]]
  tr <- tr_d %>% select(-!! y_label)
  
  te_y <- (te_d %>% select(!! y_label))[[1]]
  te <- te_d %>% select(-!! y_label)
  
  message('Train and Test data created..')
  
  
  if(is.null(lossfunction)){
    lossfunction <- 'RMSE'
    if(is.factor(tr_y)){
      lossfunction <- 'Logloss'
      if(length(levels(tr_y))>2){
        lossfunction <- 'MultiClass'
      }
      tr_y <- as.double(tr_y)-1
      te_y <- as.double(te_y)-1
    }
  }
  
  message("Loss function :", lossfunction)
  
  if(!is.null(grid_len)){
    .grid <- data.frame(depth = rep(4L, grid_len),
                       learning_rate = runif(grid_len, min = 1e-3, max = 1L),
                       iterations = rep(1000L, grid_len),
                       l2_leaf_reg = sample(c(1e-1, 1e-3), grid_len, replace = TRUE),
                       rsm = sample(c(1., 0.9, 0.8, 0.7), grid_len, replace = TRUE))
  }else{
    .grid <- NULL
  }
  
  tr_pool <- catboost.load_pool(data = tr,label = unlist(tr_y),
                                cat_features = cat_features)
  te_pool <- catboost.load_pool(data = te, label = unlist(te_y),
                                cat_features = cat_features)
  message('Catboost data created')
  
  if(lossfunction=='RMSE'){
    catopt_fit <- function(depth,learning_rate,iterations,
                           l2_leaf_reg,rsm){
      
      message('Params from Optimization. Running Models')
      
      model_fit <- catboost.train(
        learn_pool = tr_pool,
        test_pool = te_pool,
        params = list(loss_function=lossfunction,
                      eval_metric=evalmetric,
                      depth=depth,
                      learning_rate=learning_rate,
                      iterations=iterations,
                      l2_leaf_reg=l2_leaf_reg,
                      rsm=rsm,
                      border_count=border_count)
      )
      eval_pred <- catboost.predict(model_fit,te_pool)
      message('Eval prediction is Done')
      score = Metrics::rmse(te_y,eval_pred)
      
      list(Score=-score,Pred=-score)
    }
  }else{
    catopt_fit <- function(depth,learning_rate,iterations,
                           l2_leaf_reg,rsm){
      
      message('Params from Optimization. Running Models')
      
      model_fit <- catboost.train(
        learn_pool = tr_pool,
        test_pool = te_pool,
        params = list(loss_function=lossfunction,
                      eval_metric=evalmetric,
                      depth=depth,
                      learning_rate=learning_rate,
                      iterations=iterations,
                      l2_leaf_reg=l2_leaf_reg,
                      rsm=rsm,
                      border_count=border_count)
      )
      eval_pred <- catboost.predict(model_fit,te_pool)
      message('Eval prediction is Done')
      score = Metrics::auc(te_y,eval_pred)
      
      list(Score=score,Pred=score)
    }
  }
  message('Running Bayesian Optimization...')
  opt_res <- BayesianOptimization(catopt_fit,
                                  bounds = list(
                                    depth=depth_range,
                                    learning_rate=eta_range,
                                    iterations=iter_range,
                                    l2_leaf_reg=l2_reg_range,
                                    rsm=rsm_range
                                  ),
                                  init_points = init_points,
                                  init_grid_dt = .grid,
                                  n_iter = n_iter,
                                  acq = acq,
                                  kappa = kappa,
                                  eps = eps,
                                  kernel = optkernel,
                                  verbose = TRUE
  )
  
  
  return(opt_res)
  
}


moddd <- cat_opt(tr_dd,va_dd,'Preferred_Theme','AUC',grid_len = 10)  

modd2 <- cat_opt(tr_dd,va_dd,'Beauty',evalmetric='MSLE',grid_len = 10) 
