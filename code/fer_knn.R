# Install packages
install.packages("BiocManager")         
BiocManager::install("EBImage")           
install.packages("class")              
install.packages("caret")                
install.packages("ggplot2")

# Load packages 
library(EBImage)
library(class)
library(caret)
library(ggplot2)

#Load and Preprocess Images
image_dir <- "enter_path_of_dataset_folder" 

# List all TIFF image files in the folder
image_files <- list.files(image_dir, pattern = ".tiff$", full.names = TRUE)

# Initialize containers for image data and labels
image_list <- list()
labels <- c()

# Loop through images to read and preprocess
for (file in image_files) {
  img <- readImage(file)               
  img <- resize(img, 64, 64)           
  img <- channel(img, "gray") 
  vec <- as.vector(img@.Data)  
  image_list[[length(image_list) + 1]] <- vec 
# Emotion label extraction
fname <- basename(file)
parts <- unlist(strsplit(fname, "\\."))   
code <- substr(parts[2], 1, 2)            
emotion <- switch(code,
                  "AN" = "Anger",
                  "DI" = "Disgust",
                  "FE" = "Fear",
                  "HA" = "Happy",
                  "NE" = "Neutral",
                  "SA" = "Sad",
                  "SU" = "Surprise",
                  "Unknown")

labels <- c(labels, emotion)
}

# Create Feature Matrix and Labels
image_matrix <- do.call(rbind, image_list) 
labels <- as.factor(labels)  

# Train-Test Split
set.seed(123)
split_index <- createDataPartition(labels, p = 0.8, list = FALSE) 
train_data <- image_matrix[split_index, ]
test_data <- image_matrix[-split_index, ]
train_labels <- labels[split_index]
test_labels <- labels[-split_index]

# Apply KNN Algorithm
k <- 3
predicted_labels <- knn(train = train_data, test = test_data, cl = train_labels, k = k)

# Evaluate Model Performance
conf <- confusionMatrix(predicted_labels, test_labels)
print(conf)

#	Visualize Accuracy for Different K Values
accuracy_list <- c()
k_vals <- 1:15
for (k in k_vals) {
  preds <- knn(train = train_data, test = test_data, cl = train_labels, k = k)
  acc <- mean(preds == test_labels)
  accuracy_list <- c(accuracy_list, acc)
}
# Create data frame for plotting
df <- data.frame(k = k_vals, accuracy = accuracy_list)
# Plot accuracy vs k
ggplot(df, aes(x = k, y = accuracy)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  labs(title = "Accuracy vs K (KNN)",
       x = "K Value", y = "Accuracy") +
  theme_minimal()
