########################################
#### UI ################################
########################################

ui <- shinyUI(fluidPage(
    
    titlePanel("Corpus Explorer"),
    
    tabsetPanel(
        tabPanel(title = "Document-Topic Network", 
            fluidRow(
                column(8,
                    verticalLayout(fluid = T,
                        forceNetworkOutput("force", width = '100%', height = config$frameHeight),
                        fluidRow(#sliderInput(inputId = "size", label = "Size", value = 0.5, min = 0.1, max = 1, step = .1),
                            column(2,sliderInput(inputId = "opacity", label = "Opacity", value = 0.8, min = 0.1, max = 1, step = .1)),
                            column(4,sliderInput(inputId = "networkClusters", label = "Number of Clusters", value = 12, min = 1, max = 20, 
                                                         step = 1, width = '100%')),
                            column(2,sliderInput(inputId = "topicNetworkThreshold", label = "Topic Threshold", value = .25, min = 0, 
                                                         max = 1, step = .02))
                        )
                    )
                ),
                column(4,
                       fluidRow(
                           column(6,selectInput(inputId = 'documents', label = 'Documents', choices = sort(rownames(docStats)), 
                                            selected = sort(rownames(docStats)), multiple = T, selectize = F, size = 20)),
                           column(6,selectInput(inputId = 'topics', label = 'Topics' , choices = unname(topicLabels),
                                            selected = unname(topicLabels), multiple = T, selectize = T))
                       )
                )
            )
        ),
        tabPanel(title = "Document Dendrogram",
            verticalLayout(fluid = T,
                dendroNetworkOutput("dendrogram",width = '100%', height = config$frameHeight),
                fluidRow(
                    column(4,sliderInput(inputId = "dendroFontSize", label = "Font Size", value = 5, min = .1, max = 12, step = .1)),
                    column(8,sliderInput(inputId = "dendroClusters", label = "Number of Clusters", value = 20, min = 1, max = 50, 
                                                              step = 1, width = '100%'))
                )
            )
        ),
        tabPanel(title = "Document Stats",
            fluidRow(
                column(6,
                    selectInput(inputId = 'singleDoc', label = 'Document', choices = sort(rownames(docStats))),
                    h2("Topic Distribution: "),
                    plotOutput("docTopicBars"), 
                    h2("Topic Terms:"),
                    dataTableOutput("docTopics")),
                column(3,
                    h2("Significant terms by TF-IDF:"),
                    dataTableOutput("docTopTerms")),
                column(3,
                    h2("Most Similar Docs:"),
                    dataTableOutput("similarDocs"))
                 )
        ),
#         tabPanel(title = "Document Comparison",
#             selectInput(inputId = 'doc1', label = 'Document 1', choices = sort(rownames(docStats))),
#             selectInput(inputId = 'doc2', label = 'Document 2', choices = sort(rownames(docStats)))
#         ),
#         tabPanel(title = "Corpus-level Document Stats",
#         ),
        tabPanel(title = "Upload New Documents",
            fileInput(inputId = "newDocs", label = "File", multiple = T),
            dataTableOutput(outputId = "newDocs")
        )
    )
))
