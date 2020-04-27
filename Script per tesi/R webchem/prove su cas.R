# importo i cas number e la libreria webchem
library('readr')
cas <- read_csv("C:/Users/Simone/Desktop/TESI -- lavoro/Script per tesi/dataset/only_cas.csv", col_types = cols(X1 = col_skip()))



library('webchem')
?cir_query
?ci_query


to_smiles <- function(num){
  r <- cir_query(num, 'smiles')
  return(r[[1]])
}


#FROM DF TO MATRIX
cas_matrix <- as.matrix(cas$cas)

#WE APPLY THE FUNTION TO EACH ELEMENT
smiles <- apply(cas_matrix, 1, to_smiles)


df <- as.data.frame(smiles)

#MERGE AND SAVE
cas_to_smiles <- data.frame(cas,df)
write.csv(cas_to_smiles,'cas_to_smiles.csv')

cas_smiles <- read_csv("C:/Users/Simone/Desktop/TESI -- lavoro/Script per tesi/dataset/cas_to_smiles.csv", col_types = cols(X1 = col_skip()))



# - - - - - - - - - - - - - - - - - - - - - - - - - - - -- -

# peso molecolare
# si dovrebbe chiamare molWeight
massWeight <- function(num){
  mw <- cir_query(num, representation = 'mw')
  return(mw[[1]])
}

mw <- apply(cas_matrix, 1, massWeight)

df <- as.data.frame(mw)

cas_mw = data.frame(cas,df)
write.csv(cas_mw, 'cas_mw.csv')


# kow index
kow <- function(num){
  k <- cir_query(num, representation = 'xlogp2')
  return(k[[1]])
}

k <- apply(cas_matrix,1, kow)

# water solubility sta in pan_query()
# non li trovaaaaaa....
