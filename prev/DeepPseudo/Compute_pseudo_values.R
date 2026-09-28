library(prodlim)
library(survival)

pseudo_CRA <- function(data, evaltime){

  time   <- data[, "time"]
  status <- data[, "status"]

  # ensure integer status
  status <- as.integer(status)
  unq_status <- sort(unique(status[status != 0]))
  
  K <- length(unq_status)
  n <- length(time)
  T <- length(evaltime) 

  # allocate using K (not length(unique(status))-1)
  pseudo_value <- array(0, dim = c(n, K, T))

  # If this client has zero events, return all zeros (no stopping)
  if (sum(status != 0) == 0 || K == 0) {
    return(pseudo_value)
  }

  f <- prodlim(Hist(time, status) ~ 1) 
  for (k in seq_len(K)) {
    pv <- jackknife(f, times = evaltime, cause = unq_status[k])
    pv <- as.matrix(pv)

    # common: jackknife returns T x n, transpose to n x T
    if (nrow(pv) == T && ncol(pv) == n) pv <- t(pv)

    if (nrow(pv) != n || ncol(pv) != T) {
      stop(sprintf("jackknife returned %dx%d; expected %dx%d", nrow(pv), ncol(pv), n, T))
    }

    pseudo_value[, k, ] <- pv
  }

  return(pseudo_value)
} 
