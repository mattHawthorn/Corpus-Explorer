########################################
#### Server ############################    
########################################
server <- function(input, output) {
    
    # generate topic bar chart for single doc
    makeDocTopics <- reactive({
        topicBars <- docTopics[[input$singleDoc]]
        topicBars <- topicBars[order(topicBars$weight, decreasing = T),]
        topicBars$terms <- sapply(topicBars$topic, function(topic){
            paste0(topic,': ',paste0(topicTerms[[as.character(topic)]]$token[1:20], collapse = ', '))
        })
        return(topicBars)
    })
    
    # Display topic bar chart
    output$docTopicBars <- renderPlot({
        topicBars <- makeDocTopics()
        
        barplot(height = topicBars$weight, names.arg = topicBars$topic, 
                main = paste0(input$singleDoc,"\nLength (significant tokens): ", docStats[input$singleDoc,'length'], 
                              "\nTopic entropy (normalized to 1): ", 
                              round(docStats[input$singleDoc,'topicEntropy']/log(length(names(topicTerms))), 3)),
                col = sapply(topicBars$weight, function(weight) heatColors[round(1 + 99*weight)]), ylim = c(0,1))
    })

    # Display document-topic table    
    output$docTopics <- renderDataTable({
        topicBars <- makeDocTopics()
        
        return(topicBars)
    })
    
    # Display document top terms
    output$docTopTerms <- renderDataTable({
        docTopTerms <- docTerms[[input$singleDoc]]
        
        return(docTopTerms)
    })
    
    # Display most similar docs
    output$similarDocs <- renderDataTable({
        similarDocs <- data.frame(docDist[,input$singleDoc])
        names(similarDocs) = c('JS-divergence')
        similarDocs$name = rownames(similarDocs)
        similarDocs <- similarDocs[order(similarDocs[['JS-divergence']])[2:21],]
        
        return(similarDocs)
    })
    
    # Doc-Topic bipartite network
    output$force <- renderForceNetwork({
        # convert from topic labels (string of terms) to topic ID's
        topics <- names(topicLabels[topicLabels %in% input$topics])
        
        # get cluster labels for docs
        docLabels <- labelDocClusters(docClusters = docClusters, numClusters = input$networkClusters)
        
        docTopicNetworkLinks <- docTopicNetworkAllLinks[doc %in% input$documents & topic %in% topics & weight >= input$topicNetworkThreshold]
        docTopicNetworkNodes <- makeDocTopicNetworkNodes(docNames = input$documents, topicNames = topics,
                                                         docStats = docStats, docLabels = docLabels, topicTerms = topicTerms)
        docTopicNetworkLinks$doc <-  match(docTopicNetworkLinks$doc,rownames(docTopicNetworkNodes)) - 1
        docTopicNetworkLinks$topic <-  match(docTopicNetworkLinks$topic,rownames(docTopicNetworkNodes)) - 1
        
        forceNetwork(Links = docTopicNetworkLinks, Nodes = docTopicNetworkNodes, Source = "doc", Target = 'topic', Value = "weight",
                     NodeID = "name", Group = "cluster", Nodesize = "size",
                     zoom = T, fontSize = 6, bounded = F, linkDistance = 1, colourScale = 'd3.scale.category20b()',
                     #radiusCalculation = JS("Math.sqrt(d.nodesize)"),
                     #linkDistance = JS("function(d){ return 1/(d.value + 1);}"),
                     #linkWidth = JS("function(d){ return Math.sqrt(d.value);}"),
                     opacity = input$opacity, width = config$frameWidth, height = config$frameHeight)
    })
    
    # Display document dendrogram from hierarchical clustering
    output$dendrogram <- renderDendroNetwork({
        
        docLabels <- labelDocClusters(docClusters = docClusters, numClusters = input$dendroClusters)
        nodeColors = sapply(docClusters$labels, function(docName){ docLabel = docLabels[docName,'cluster'] %% length(nodePalette)
        if(docLabel == 0){ docLabel = length(nodePalette)}
        nodePalette[docLabel]},
        USE.NAMES = F)
        
        dendroNetwork(hc = docClusters, width = config$frameWidth, height = config$frameHeight, zoom = T, linkType = 'diagonal', 
                      opacity = 0.8, textColour = nodeColors, 
                      #nodeColour = 'blue', 
                      fontSize = input$dendroFontSize, margins = list(top=20,bottom=20,left=20,right=300))
    })
    
    
    ingest <- reactive({
        newDocTable = input$newDocs
#         if(!is.null(newDocTable)){
#             # Write the gensim model output data for the new docs:
#             python.call("ingestNewDocs", newFiles = newDocTable$name, newPaths=newDocTable$datapath)
#             # Read in the output:
#             ingestNewDocs(docDir = 'data/new')
#             # parent link set; links for network displays will be subset on this
#             docTopicNetworkAllLinks <- rbindlist(docTopics,idcol = 'doc')
#             setkeyv(docTopicNetworkAllLinks, c('doc','topic'))
#         }
        return(newDocTable)
    })
    
    # Show which documents were ingested
    output$newDocs <- renderDataTable({
        newDocTable <- ingest()
        return(newDocTable)
    })
    
    
}



