data {
  int Z; // Number of pairwise predictors
  int<lower=0> N; // Number of individuals
  int<lower=2> M;//  Number of pairs
  int<lower=0> L;// Number of census tracts
  
  matrix [M, Z] XP; //Pairwise predictors

  int<lower=0,upper=1> y[M]; // Pairwise linkage between individuals in pairs
  int<lower=1,upper=N> a[M]; // Indices of individual a in each of M pairs
  int<lower=1,upper=N> b[M]; // Indices of individual b in each of M pairs
  int<lower=1,upper=L> r[M]; // Indices of census tract r in each of M pairs
  int<lower=1,upper=L> s[M]; // Indices of census tract s in each of M pairs


}

parameters {
  real mu_logit; //population average
  vector[N] alpha; //individual random effects
  vector[L] alpha_tract; //tract level random effects
  vector[Z] beta; //pairwise covariates
  real<lower=0> sigma; //variance of individual random effects
  real<lower=0> sigma_tract; //variance of tract random effects
}


model {
    //priors
    mu_logit ~ normal(-7, 3);
    alpha ~ normal(0, sigma);
    alpha_tract ~ normal(0, sigma_tract);
    sigma ~ exponential(1);
    sigma_tract ~ exponential(1);
    beta ~ normal(0, 2);
  
   vector[M] eta = mu_logit
                  + alpha[a] + alpha[b]
                  + alpha_tract[r] + alpha_tract[s]
                 + XP * beta; // Dot-product over all pairs
                  
  y ~ bernoulli_logit(eta);
}



