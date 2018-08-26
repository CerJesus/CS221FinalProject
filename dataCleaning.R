library(dplyr)

# ====
# Cleaning the training data
# ====

data = read.csv("data/train.csv")
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
# ====

# ====
# Adding neighborhood-specific data
# ====

unique(data$Neighborhood)

zed = data %>% 
  group_by(Neighborhood) %>% 
  summarise(num = n())

neigh = read.csv("data/neighborhood_data.csv") %>% 
  filter(X <= 25) %>% 
  select(-X.1)

neigh = neigh %>% 
  mutate(Median_House_Income = as.numeric(gsub(",", "", Median_House_Income))/1000) %>% 
  mutate(Median_House_Price = as.numeric(gsub(",", "", Median_House_Price))/1000)

# Mean house price
means = data %>% 
  group_by(Neighborhood) %>% 
  summarise(Mean_Price = mean(SalePriceThousands))
View(means)

# Standard deviation of house price
std = data %>% 
  group_by(Neighborhood) %>% 
  summarise(Std_Price = sd(SalePriceThousands))
View(std)

# Median of house Price
med = data %>% 
  group_by(Neighborhood) %>% 
  summarise(Median_Price = median(SalePriceThousands))
View(med)

neigh_data = merge(means, std, by="Neighborhood")
neigh_data = merge(neigh_data, med, by="Neighborhood")
neigh_data = merge(neigh, neigh_data, by="Neighborhood")
neigh_data = neigh_data %>%
  select(-Median_House_Price) %>% 
  select(-X)
View(neigh_data)
write.csv(neigh_data, "data/neighborhood_data_final.csv")

# ====

# ====
# Combining the two data tables
# ====

zed = merge(data, neigh_data, by="Neighborhood")
write.csv(zed, "data/data_house_and_neighborhood.csv")

# ====
# Removing some features
# ====

data = read.csv("data/train_updated.csv")
View(data)


data = data %>% 
  select(-LotShape) %>% 
  select(-LandContour) %>% 
  select(-Utilities) %>% 
  select(-LotConfig) %>% 
  select(-LandSlope) %>% 
  select(-RoofStyle) %>%
  select(-MasVnrType) %>%
  select(-MasVnrArea) %>%
  select(-BsmtExposure) %>%
  select(-Foundation) %>%
  select(-HeatingQC) %>%
  select(-CentralAir) %>%
  select(-FireplaceQu) %>%
  select(-GarageFinish) %>%
  select(-PavedDrive) %>%
  select(-MiscFeature) %>%
  select(-MiscVal)
  
write.csv(data, "data/train_trimmed.csv")



summary = data %>% summarise_all(funs(n_distinct(.)))
  