
library(dplyr)

setwd("~/Desktop/CS221/CS221FinalProject")

data = read.csv("train.csv")
View(data)

data = data %>% 
  mutate(HouseAge = YrSold - YearBuilt) %>% 
  mutate(GarageAge = YrSold - GarageYrBlt) %>% 
  mutate(RemodAddAge = YrSold - YearRemodAdd) %>% 
  mutate(SalePriceThousands = SalePrice/1000.0) %>% 
  select(-SalePrice) %>% 
  select(-YearBuilt) %>% 
  select(-YearRemodAdd) %>% 
  select(-GarageYrBlt) %>% 
  mutate(Id =  1) %>% 
  select(-Street) %>% 
  mutate(Utilities = ifelse(Utilities == 'AllPub', 1, 0))

data = data %>% 
  mutate(LotFrontage = ifelse(is.na(LotFrontage), 0, LotFrontage)) %>% 
  mutate(GarageAge = ifelse(is.na(GarageAge), 0, GarageAge)) %>%
  mutate(MasVnrArea = ifelse(is.na(MasVnrArea), 0, MasVnrArea))

unique(data$Neighborhood)

zed = data %>% 
  group_by(Neighborhood) %>% 
  summarise(num = n())

write.csv(zed, "neighborhoods_count.csv")

write.csv(data, "train_updated.csv")
  