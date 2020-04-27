library(readr)
library('webchem')

cas_na_mw <- read_csv("C:/Users/Simone/Desktop/TESI -- lavoro/Script per tesi/dataset_prova/cas_na_mw.csv", col_types = cols(X1 = col_skip()))

molWeight <- function(num){
  mw <- cir_query(num, representation = 'mw')
  return(mw[[1]])
}

cas_matrix <- as.matrix(cas_na_mw$cas)

mw <- apply(cas_matrix, 1, molWeight)

df <- as.data.frame(mw)

cas_find_mw <- data.frame(cas_na_mw, df)
write.csv(cas_find_mw, 'cas_find_mw.csv')
