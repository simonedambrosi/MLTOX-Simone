#IMPORTING LIBRARY
library("webchem")

#READING THE CSV FILE
cas = read.csv('cas_number.csv')
cas <- cas[c("test_cas")]

# FUNCTION WHICH TAKE THE SMILES
to_smiles <- function(num){
  r <- cir_query(num, 'smiles')
  return(r[[1]])
}


#FROM DF TO MATRIX
cas_matrix <- as.matrix(cas$test_cas)

#WE APPLY THE FUNTION TO EACH ELEMENT
smiles <- apply(cas_matrix, 1, to_smiles)


df <- as.data.frame(smiles)

#MERGE AND SAVE
cas_to_smiles <- data.frame(cas,df)
write.csv(cas_to_smiles,'cas_to_smiles.csv')