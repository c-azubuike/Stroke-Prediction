### install and load the necessary packages
install.packages(c('readr','tidyr','dplyr','skimr','ggplot2','corrplot','caret',
                   'randomForest'))
library(readr)
library(tidyr)
library(dplyr)
library(skimr)
library(ggplot2)
library(corrplot)
library(caret)
library(randomForest)
library(pROC)

### load dataset and view the first and last few rows
main_data <- read_csv('healthcare-dataset-stroke-data.csv')
head(main_data,10)
tail(main_data,10)

## split the dataset into "test" and "training" before pre-processing
set.seed(45)
trainIndex <- createDataPartition(main_data$stroke, p = 0.8, list = FALSE)
trainData <- main_data[trainIndex,]
testData <- main_data[-trainIndex,]
cat('Rows of training data:',nrow(trainData),'\n" " test data:',nrow(testData))

### handle missing data and duplicate rows
trainData[duplicated(trainData),] # checks for duplicate rows
trainData %>% mutate(na_counts = rowSums(across(everything(), ~ is.na(.))))

## first operation found no NA values but bmi has 'N/A'
# convert the N/A to NA and replace the NA with the median bmi value
trainData <- trainData %>% mutate(bmi = ifelse(bmi == 'N/A',NA,bmi))
trainData %>% mutate(na_counts = rowSums(across(everything(), ~ is.na(.))))

## perform median imputation on trainData
trainData$bmi <- as.numeric(trainData$bmi)# first change column to numeric
trainData <- trainData %>% mutate(bmi = ifelse(is.na(bmi),
                                               median(bmi, na.rm = T),bmi))
trainData %>% mutate(na_counts = rowSums(across(everything(), ~ is.na(.))))

### explore the dataset for patterns
summary(trainData)
skim(trainData)

## distribution of numerical columns
hist(trainData$age)
boxplot(trainData$bmi)
boxplot(trainData$avg_glucose_level)

## stroke trends across gender
ggplot(trainData, aes( x = gender, fill = factor(stroke))) +
  geom_bar( position = 'dodge') +
  scale_fill_manual(name = 'Stroke Status',
                    values = c('0' = 'blue','1' = 'gold'),
                    labels = c('absent','present')) + 
  geom_text(stat = 'count', aes(label = after_stat(count)),
            position = position_dodge2(width =0.9 ), vjust = -0.2)+
  labs(title = 'Stroke counts by gender')+ theme_minimal()

## stroke variation by age
ggplot(trainData, aes(x = factor(stroke), y = age)) + geom_boxplot() +
  labs(title = ' Variation in age across stroke status',
       x ='stroke status',y = 'age') + theme_minimal()

## stroke trends and smoking_status
ggplot(trainData, aes(x = smoking_status, fill = factor(stroke)))+
  geom_bar(position = 'dodge')+scale_fill_manual(name = 'Stroke Status',
                              values = c('0'='blue','1'='gold'),
                              labels = c('absent','present'))+
  geom_text(stat = 'count', aes(label = after_stat(count)),
            position = position_dodge2(width = 0.9), hjust = -0.2) +
  labs(title = 'Smoking status and Stroke Occurrence', x = 'count',
       y = 'smoking status') + theme_minimal() + coord_flip()

## stroke status by heart disease
ggplot(trainData, aes(x = factor(heart_disease), fill = factor(stroke)))+
  geom_bar(position = 'dodge')+scale_fill_manual(name = 'Stroke Status',
                              values = c('0'='blue','1'='gold'),
                              labels = c('absent','present'))+
  geom_text(stat = 'count',aes(label = after_stat(count)), 
            position = position_dodge2(width =  0.9), vjust = -0.2) +
  labs(title = ' Stroke status by Heart Disease', x = 'heart disease',
       y = 'count') + theme_minimal()

### convert character variables with minimal unique values to factors
## do the same for both training and test data
skim(trainData) # use skim() as a guide
trainData$gender <- as.factor(trainData$gender)
trainData$ever_married <- as.factor(trainData$ever_married)
trainData$work_type <- as.factor(trainData$work_type)
trainData$Residence_type <- as.factor(trainData$Residence_type)
trainData$smoking_status <- as.factor(trainData$smoking_status)

### Change similar columns in the test dataset
## bmi will not be modified thus far to prevent possible data leakage
skim(testData)
testData$gender <- as.factor(testData$gender)
testData$ever_married <- as.factor(testData$ever_married)
testData$work_type <- as.factor(testData$work_type)
testData$Residence_type <- as.factor(testData$Residence_type)
testData$smoking_status <- as.factor(testData$smoking_status)
str(testData) # validate the conversion

### analysis phase using training data
## earlier box plots of bmi and avg glucose level showed outliers
# check range and apply log transformation to both columns
range(trainData$bmi)
range(trainData$avg_glucose_level)
trainData <- trainData %>% mutate(log_bmi = log(bmi),
    log_avg_glucose_level = log(avg_glucose_level))
range(trainData$log_bmi)
range(trainData$log_avg_glucose_level)

### assess the range of age column before re scaling other numerical columns
## 'heart_disease' and 'hypertension' are already normalized
range(trainData$age)

### apply min-max re scaling technique
trainData <- trainData %>% mutate(
scaled_age = (age - min(age))/(max(age) - min(age)),
scaled_log_avg_glucose_level = (log_avg_glucose_level - 
   min(log_avg_glucose_level))/
  (max(log_avg_glucose_level) - min(log_avg_glucose_level)),
scaled_log_bmi = ((log_bmi - min(log_bmi))/(max(log_bmi) - min(log_bmi)))
)

skim(trainData) # summary of trainData

### drop the 'id' column from both the train and test Data
trainData <- trainData %>% select(-id)
testData <- testData %>%  select (-id)

names(trainData)
names(testData)

### statistical tests
t.test(age ~ stroke, data = trainData)
t.test(bmi ~ stroke, data = trainData)
t.test(avg_glucose_level ~ stroke, data = trainData)

chisq.test(table(trainData$hypertension, trainData$stroke))
chisq.test(table(trainData$heart_disease, trainData$stroke))
chisq.test(table(trainData$gender, trainData$stroke)) #no association
chisq.test(table(trainData$ever_married, trainData$stroke))
chisq.test(table(trainData$work_type, trainData$stroke))
chisq.test(table(trainData$Residence_type, trainData$stroke))# no association
chisq.test(table(trainData$smoking_status, trainData$stroke))

## key variables from statistical tests :age,bmi,avg_glucose_level,hypertension,
# heart_disease, ever_married, work_type, smoking_status

### apply changes made to train dataset to the test dataset
class(testData$bmi) # verify data type
testData <- testData %>% mutate(bmi = ifelse(bmi == 'N/A', NA, bmi))
testData[is.na(testData$bmi),]

## Perform median imputation for bmi in testData using median from trainData
testData$bmi <- as.numeric(testData$bmi)
trainData_median_bmi <- median(trainData$bmi, na.rm = TRUE)
testData$bmi <- ifelse(is.na(testData$bmi), trainData_median_bmi, testData$bmi)
testData[is.na(testData$bmi),] # confirm imputation

### apply the log tranformation to the test dataset
range(testData$bmi)
range(testData$avg_glucose_level)
testData$log_bmi <- log(testData$bmi)
testData$log_avg_glucose_level <- log(testData$avg_glucose_level)

# Apply Min-Max scaling to test data using min and max values from train data
testData <- testData %>% mutate(
scaled_age = (age - min(trainData$age)) / 
  (max(trainData$age) - min(trainData$age)),
scaled_log_avg_glucose_level = (log_avg_glucose_level - 
                                min(trainData$log_avg_glucose_level)) /
(max(trainData$log_avg_glucose_level) - min(trainData$log_avg_glucose_level)),
scaled_log_bmi = (log_bmi - min(trainData$log_bmi)) / 
  (max(trainData$log_bmi) - min(trainData$log_bmi)))

### remove unscaled and irrelevant columns from trainData
trainData <- trainData %>% 
select(-c(gender, Residence_type, age, avg_glucose_level, bmi, 
          log_bmi, log_avg_glucose_level))

### remove unscaled and irrelevant columns from testData
testData <- testData %>% 
  select(-c(gender, Residence_type, age, avg_glucose_level, bmi, 
            log_bmi, log_avg_glucose_level))

### check if both datasets have the similar features
identical(names(trainData),names(testData))
str(trainData)
str(testData)

## change hypertension, heart_disease and stroke to factor in both data sets
trainData <- trainData %>%
  mutate(
    hypertension = factor(hypertension, levels = c(0, 1)),
    heart_disease = factor(heart_disease, levels = c(0, 1)),
    stroke = factor(stroke, levels = c(0, 1))
  )

testData <- testData %>%
  mutate(
    hypertension = factor(hypertension, levels = c(0, 1)),
    heart_disease = factor(heart_disease, levels = c(0, 1)),
    stroke = factor(stroke, levels = c(0, 1))
  )

### ensure that the levels of other factor attributes in both data sets
## are aligned
testData$ever_married <- factor(testData$ever_married,
                                levels = levels(trainData$ever_married))
testData$work_type <- factor(testData$work_type,
                                levels = levels(trainData$work_type))
testData$smoking_status <- factor(testData$smoking_status, 
                                levels = levels(trainData$smoking_status))

# validate the structure of both datasets
str(trainData)
str(testData)

### separate the target variable "stroke" from the rest of the features
X_train <- trainData %>% select(-stroke)  # select everything but stroke
y_train <- trainData$stroke

## do the same to testData
X_test <- testData %>% select(-stroke)
y_test <- testData$stroke

### Train a stroke prediction model
table(trainData$stroke) # checks distribution of stroke and noStroke
table(testData$stroke)

## since dataset is heavily imbalanced, use a weighted random forest model
# this will assign more weight to the minority class
# count of '0' is 966 and count of '1' is 56 in testData so '1' is 17 times less
model <- randomForest(stroke ~., data = trainData, importance = T,
      weights = ifelse(trainData$stroke == 1,17,1)) # stroke has a weight of 17

### make a prediction about stroke status using columns (except stroke) in
## test dataset
# this will create probabilites for each row and store it in a vector
prediction <- predict(model, newdata = X_test, type = 'prob')
prob_class_stroke <- prediction[,2]

# convert probabilities to binary
# 0.3 has the best balance between sensitivity and specificity
prediction_binary <- ifelse(prob_class_stroke > 0.3, 1, 0)

### create a confusion matrix to assess the performance of the model
confusionMatrix(factor(prediction_binary),factor(y_test))

### Plot feature importance to see which influences the model the most
## use the variable importance plot function from randomForest
importance(model)
varImpPlot(model, main = "Feature Importance of Stroke Prediction Model")

### calculate how well the model can distinguish between the two stroke classes
## y_test contains two stroke levels (0 and 1)
# prob_class_stroke contains predicted prob of having stroke('1')
roc_comparison <- roc(y_test, prob_class_stroke)
plot(roc_comparison)
auc(roc_comparison) # area under the curve is greater than 0.80



