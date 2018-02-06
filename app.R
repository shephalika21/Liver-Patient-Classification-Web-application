library(shiny)
library(lattice)
library(ROCR)
library(ggplot2)
library(gplots)

# Define UI for application
ui <- fluidPage(
  
  # Application title
  titlePanel("Classification Models on ILPD Data Set"),
  
  tabsetPanel(
    navbarPage("ILPD",
               tabPanel("Raw Data", sidebarPanel(
                 p(strong("Original data")),
                 numericInput("obs", "Number of observations to view:", 10)
                 
               ),
               mainPanel(
                 tabsetPanel(
                   tabPanel("Results",
                            tableOutput("RawTable"),
                            plotOutput("Raw"),
                            verbatimTextOutput("summary")
                            #htmlOutput("Text")
                   )
                 )
               )),
               tabPanel("Classification Algorithms",
                        
                        sidebarPanel(
                          radioButtons("type", "Algorithm type:",
                                       c("Decision Trees" = "DT",
                                         "Naive Bayes" = "NB",
                                         "Logistic Regression" = "LR",
                                         "Random forest" = "RF",
                                         "K-NN" = "knn"
                                         
                                       ))
                        ),
                        mainPanel(
                          tabsetPanel(
                            tabPanel("Results",
                                     plotOutput("typeControls"),
                                     #htmlOutput("header")
                                     verbatimTextOutput("text"),
                                     tableOutput("Table")
                            )
                          )
                        )
               ),
               tabPanel("Analysis",sidebarPanel(
                 p(strong("AUC for Decision Tree:")),
                 textOutput("acc"),
                 p(strong("AUC for Naive Bayes:")),
                 textOutput("acc1"),
                 p(strong("AUC for Logistic Regression:")),
                 textOutput("acc2"),
                 p(strong("AUC for Random Forest:")),
                 textOutput("acc3"),
                 p(strong("Based on the above AUC's we can say that Logistic Regression is the best model for Indian Liver Patient DataSet Prediction as it has AUC of:")),
                 textOutput("acc4")
               ),
               mainPanel(
                 tabsetPanel(
                   tabPanel("Results",
                            plotOutput("AllGraphs"),
                            plotOutput("AllGraph1"),
                            plotOutput("AllGraph2"),
                            plotOutput("AllGraph3")
                            
                   )
                 )
               ))
               
    ))
)

# Define server logic and the models
server <- function(input, output) {
  
  output$typeControls <- renderPlot({
    dataFinal <- generatePlot(input$type)
    
    
  })
  
   generatePlot <- function(type){
    set.seed(123)
    
    library(rpart)
    library(caret)
    library(rpart.plot)
    library(ROCR)
    library(psych)
    library(e1071)
    library(arules)
    
    
    dataset = read.csv('Indian Liver Patient Dataset (ILPD).csv', header=TRUE)
    dataset = dataset[-2]
    colSums(is.na(dataset))
    dataset$ag <- ifelse(is.na(dataset$ag), mean(dataset$ag, na.rm=TRUE), dataset$ag)
    colSums(is.na(dataset))
    #Creating factor levels for label
    dataset$label <- factor(dataset$label , levels =  c(0,1))
    
    # Splitting the dataset into the Training set and Test set
    # install.packages('caTools')
    library(caTools)
    split = sample.split(dataset$label, SplitRatio = 0.8)
    training_set = subset(dataset, split == TRUE)
    test_set = subset(dataset, split == FALSE)
    
    output$Raw <- renderPlot({
      plot(dataset, main="Raw data")
    })
    
    output$RawTable <- renderTable({
      head(dataset, n = input$obs)
    },striped = TRUE, bordered = TRUE,  
    hover = TRUE, spacing = 'l',  
    width = '100%', align = 'c'
    )
    
    output$summary <- renderPrint({
      
      summary(dataset)
    })
    # Feature Scaling
    training_set[-10] = scale(training_set[-10])
    test_set[-10] = scale(test_set[-10])
    
    
    # Applying PCA
    # install.packages('caret')
    library(caret)
    # install.packages('e1071')
    library(e1071)
    pca = preProcess(x = training_set[-10], method = 'pca', pcaComp = 6)
    training_set = predict(pca, training_set)
    training_set = training_set[c(2, 3,4,5,6, 1)]
    test_set = predict(pca, test_set)
    test_set = test_set[c(2, 3,4,5,6, 1)]
    
    # As seen in summary above Normalizing or standardizing is nothing but changing the magnitude of the data.
    #Normalizing always helps as its helps our algorithm to converge faster and better 
    #and if we normalize any of the attribute its important that we normalize all of 
    #the attribute otherwise variance problem occurs.
    
    type<-switch(type, 
                 "DT"={
                   #Decision Tree
                   
                   library(rpart)
                   decision_tree_model = rpart(formula = label ~ .,
                                               data = training_set, method="class")
                   
                   # Predicting the Test set results
                   y_pred = predict(decision_tree_model, newdata = test_set[-6], type = 'class')
                   
                   # Making the Confusion Matrix
                   #cm = table(test_set[, 6], y_pred)
                   cm <-confusionMatrix(y_pred, test_set[, 6])
                   
                   
                   
                   # Plotting the tree
                   
                   
                   #ROC Curve
                   pred.rocr <- predict(decision_tree_model, newdata=test_set, type="prob")[,2]
                   f.pred <- prediction(pred.rocr, test_set$label)
                   f.perf <- performance(f.pred, "tpr", "fpr")
                   #generateGraph(f.perf)
                   par(mfrow=c(1,2))
                   output$AllGraphs <- renderPlot({
                     
                     plot(f.perf, colorize=T, lwd=3,main="ROC curve for Decision Tree")
                     
                   })
                   fourfoldplot(cm$table,main="Confusion Matrix Plot")
                   
                   plot(f.perf, colorize=T, lwd=3,main="ROC curve")
                   
                   
                   output$text <- renderPrint({
                     
                     cm
                   })
                   output$Table <- renderTable({
                     
                     cm$table
                     },striped = TRUE, bordered = TRUE,  
                     hover = TRUE, spacing = 'l',  
                     width = '100%', align = 'c'
                     
                     
                   )
                   #plot(decision_tree_model)
                   #text(decision_tree_model)
                   abline(0,1)
                   auc <- performance(f.pred, measure = "auc")
                   auc@y.values[[1]]
                   output$acc <- renderText({
                     
                     
                     auc@y.values[[1]]
                   })
                   #print('DT');
                 },
                 "NB"={
                   #Naive Bayes
                   
                   library(e1071)
                   classifier = naiveBayes(x = training_set[-6],
                                           y = training_set$label, method="class")
                   
                   # Predicting the Test set results
                   y_pred = predict(classifier, newdata = test_set[-6], type="class")
                   
                   # Making the Confusion Matrix
                   # cm = table(test_set[, 6], y_pred)
                   cm1 <-confusionMatrix(y_pred, test_set[, 6])
                   output$text <- renderPrint({
                     
                     cm1
                   })
                   
                   #ROC Curve
                   pred.rocr <- predict(classifier, newdata=test_set, type="raw")[,2]# Posterior probabilities
                   f.pred <- prediction(pred.rocr, test_set$label)
                   f.perf1 <- performance(f.pred, "tpr", "fpr")
                   output$AllGraph1 <- renderPlot({
                     f.perf1 <- performance(f.pred, "tpr", "fpr")
                     plot(f.perf1, colorize=T, lwd=3 ,main="ROC curve for Naive Bayes")
                     
                   })
                   # layout(matrix(c(1,1,2,3), 2, 2, byrow = TRUE))
                   par(mfrow=c(1,2))
                   fourfoldplot(cm1$table,main="Confusion Matrix Plot")
                   plot(f.perf1, colorize=T, lwd=3 ,main="ROC curve")
                   output$Table <- renderTable({
                     
                     cm1$table},striped = TRUE, bordered = TRUE,  
                     hover = TRUE, spacing = 'l',  
                     width = '100%', align = 'c'
                     
                     
                   )
                   abline(0,1)
                   auc <- performance(f.pred, measure = "auc")
                   auc@y.values[[1]]
                   output$acc1 <- renderText({
                     
                     auc@y.values[[1]]
                   })
                   print('NB');
                 },
                 "LR"={# Fitting Logistic Regression to the Training set
                   model = glm(formula =label  ~ .,family = binomial,data = training_set)
                   
                   # Predicting the Test set results
                   prob_pred = predict(model, type = 'response', newdata = test_set[-6])
                   y_pred = ifelse(prob_pred > 0.6, 1, 0)
                   
                   # Making the Confusion Matrix
                   cm = table(test_set[, 6], y_pred > 0.6)
                   
                   cm3 <- confusionMatrix(y_pred, test_set[, 6], positive = "1")
                   
                   output$text <- renderPrint({
                     
                     cm3
                   })
                   
                   #ROC Curve
                   pred.rocr <- predict(model, newdata=test_set, type="response")
                   f.pred <- prediction(pred.rocr, test_set$label)
                   f.perf2 <- performance(f.pred, "tpr", "fpr")
                   output$AllGraph2 <- renderPlot({
                     f.perf1 <- performance(f.pred, "tpr", "fpr")
                     plot(f.perf2, colorize=T, lwd=3 ,main="ROC curve for Logistic Regression")
                     
                   })
                   # layout(matrix(c(1,1,2,3), 2, 2, byrow = TRUE))
                   par(mfrow=c(1,2))
                   fourfoldplot(cm3$table,main="Confusion Matrix Plot")
                   plot(f.perf2, colorize=T, lwd=3,main="ROC Curve")
                   output$Table <- renderTable({
                     
                     cm3$table},striped = TRUE, bordered = TRUE,  
                     hover = TRUE, spacing = 'l',  
                     width = '100%', align = 'c'
                     
                     
                   )
                   abline(0,1)
                   auc <- performance(f.pred, measure = "auc")
                   auc@y.values[[1]]
                   output$acc2 <- renderText({
                     
                     auc@y.values[[1]]
                   })
                  
                   output$acc4 <- renderText({
                     
                     auc@y.values[[1]]
                   })
                   
                   
                 },
                 "RF"={
                   # Fitting Random Forest Classification to the Training set
                   # install.packages('randomForest')
                   library(randomForest)
                   # set.seed(123)
                   random_forest_model = randomForest(x = training_set[-6],
                                                      y = training_set$label,
                                                      ntree = 100)
                   
                   # Predicting the Test set results
                   y_pred = predict(random_forest_model, newdata = test_set[-6])
                   
                   
                   # Making the Confusion Matrix
                   cm = table(test_set[, 6], y_pred)
                   
                   cm4 <- confusionMatrix(y_pred, test_set[, 6], positive = "1")
                   output$text <- renderPrint({
                     
                     cm4
                   })
                   #ROC Curve
                   pred.rocr <- predict(random_forest_model, newdata=test_set, type="prob")[,2]
                   f.pred <- prediction(pred.rocr, test_set$label)
                   f.perf3 <- performance(f.pred, "tpr", "fpr")
                   layout(matrix(c(1,1,2,3), 2, 2, byrow = TRUE))
                   plot(random_forest_model,main="Random Forest Model")
                   plot(f.perf3, colorize=T, lwd=3,main="ROC Curve")
                   output$AllGraph3 <- renderPlot({
                     f.perf1 <- performance(f.pred, "tpr", "fpr")
                     plot(f.perf3, colorize=T, lwd=3 ,main="ROC curve For Random Forest")
                     
                   })
                   fourfoldplot(cm4$table,main="Confusion Matrix Plot")
                   output$Table <- renderTable({
                     
                     cm4$table},striped = TRUE, bordered = TRUE,  
                     hover = TRUE, spacing = 'l',  
                     width = '100%', align = 'c'
                     
                     
                   )
                   abline(0,1)
                   auc <- performance(f.pred, measure = "auc")
                   auc@y.values[[1]]
                   output$acc3 <- renderText({
                     
                     auc@y.values[[1]]
                   })
                   
                 },
                   "knn"={
                     
                     library(class)
                     y_pred = knn(train = training_set[, -6],
                                  test = test_set[, -6],
                                  cl = training_set[, 6],
                                  k = 5,
                                  prob = TRUE)
                     
                     cm = confusionMatrix(y_pred, test_set[, 6], positive = "1")
                     
                     output$text <- renderPrint({
                       
                       cm
                     })
                     
                     library(ISLR)
                     library(caret)
                     ctrl <- trainControl(method="repeatedcv",repeats = 3) #,classProbs=TRUE,summaryFunction = twoClassSummary)
                     knnFit <- train(label ~ ., data = training_set, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)
                     
                     par(mfrow=c(1,2))
                     
                     knnPredict <- predict(knnFit,newdata = test_set )
                     #Get the confusion matrix to see accuracy value and other parameter values
                     confusionMatrix(knnPredict, test_set$label )
                     
                     plot(knnFit, print.thres = 0.5)
                     
                     knnPredict <- predict(knnFit,newdata = test_set , type="prob")
                     
                     output$Table <- renderTable({
                       
                       cm$table
                       
                       
                     },striped = TRUE, bordered = TRUE,  
                     hover = TRUE, spacing = 'l',  
                     width = '100%', align = 'c'
                     )
                     
                     
                     fourfoldplot(cm$table,main="ConfusionMatrix Plot")
                     print('knn');
                   }
                  
                  
                   
                
    )
    
    
    
    
  }
  z <- c('DT','NB','LR','RF')
  for(i in z){
    generatePlot(i);
  }
  
}

# Run the application 
shinyApp(ui = ui, server = server)
