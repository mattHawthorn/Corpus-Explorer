#### Load necessary packages and data ####
library(shiny)
library(networkD3)
#library(rPython)
library(jsonlite)
library(data.table)

# python code for new document ingestion/inference
#python.load("ingest_new_docs.py")

# config data: frame size in pixels
config <- jsonlite::fromJSON('config.json')


########
# Functions

# Generate document topic network nodes
makeDocTopicNetworkNodes <- function(docNames, topicNames, docStats, docLabels, topicTerms){
    
    docs = data.frame(row.names = docNames, name = docNames, stringsAsFactors = F)
    docs$size = sapply(rownames(docs), function(docName){ sqrt(docStats[docName,'length']) })
    docs$cluster = sapply(rownames(docs), function(docName){ docLabels[docName,'cluster'] })
    
    topics = data.frame(row.names = topicNames, cluster = 0, size = rep(.5*sqrt(min(docStats$length)),length(topicNames)),
                        stringsAsFactors = F)
    topics$name = sapply(rownames(topics), function(topic){topicLabels[[topic]]})

    nodes = rbind(docs,topics)
    return(nodes)
}


# Given a cluster count, label the docs by cluster using the hclust output
labelDocClusters <- function(docClusters, numClusters){
    
    temp = data.frame(row.names = docClusters$labels, cluster = rep(0,length(docClusters$labels)))
    n = nrow(temp)
    stages = n-numClusters
    for(i in 1:stages){
        join = docClusters$merge[i,]
        points = abs(join[join < 0])
        clusters = join[join > 0]
        temp$cluster[points] <- i
        temp$cluster[temp$cluster %in% clusters] <- i
    }
    
    bool = (temp$cluster == 0)
    k = sum(bool)
    if(k > 0){ temp$stage[bool] <- stages + 1:sum(bool)}
    
    order = sort(unique(temp$cluster))
    temp$cluster <- sapply(temp$cluster,function(i){match(i,order)})
    return(temp)
}

# entropy and Jensen-Shannon divergence
entropy <- function(p, q){
    p = p/sum(p)
    if(missing(q)){
        return(-1*sum(sapply(p,function(x){if(x>0){x*log(x)}else{0}})))
    } else {
        q = q/sum(q)
        bool = q > 0
        p = cbind(p[bool], q[bool])
        return(sum(apply(p,1,function(x){if(x[1]>0){x[1]*log(x[1]/x[2])}else{0}})))
    }
}

JSDivergence <- function(p1,p2){
    m = .5*(p1+p2)
    return(.5*(entropy(p1,m)+entropy(p2,m)))
}


# ingests all the corpus document data, appending if the data is new
ingestNewDocs <- function(docDir){
    newDocTerms = jsonlite::fromJSON(paste0(docDir,'/docTerms.json'),simplifyDataFrame = T,simplifyVector = T)
    newDocTopics = jsonlite::fromJSON(paste0(docDir,'/docTopics.json'),simplifyDataFrame = T,simplifyVector = T)
    newDocStats = jsonlite::fromJSON(paste0(docDir,'/docStats.json'),simplifyDataFrame = T,simplifyVector = T)
    rownames(newDocStats) <- newDocStats[['name']]
    newDocDist <- as.matrix(read.csv(paste0(docDir,'/docDist.csv'),row.names = 1, stringsAsFactors = F))
    colnames(newDocDist) <- rownames(newDocDist)
    
    if(exists('docStats',where = .GlobalEnv)){
        #assign('docStats', rbind(docStats,newDocStats), envir = .GlobalEnv)
        docStats <<- rbind(docStats,newDocStats)
    } else { 
        #assign('docStats', newDocStats, .GlobalEnv)
        docStats <<- newDocStats
    }
    if(exists('docTopics', where = .GlobalEnv)){
        #assign('docTopics', append(docTopics,newDocTopics), envir = .GlobalEnv)
        docTopics <<- append(docTopics,newDocTopics)
    } else { 
        #assign('docTopics', newDocTopics, .GlobalEnv)
        docTopics <<- newDocTopics
    }
    if(exists('docTerms', where = .GlobalEnv)){
        #assign('docTerms', append(docTerms,newDocTerms), envir = .GlobalEnv)
        docTerms <<- append(docTerms,newDocTerms)
    } else { 
        #assign('docTerms', newDocTerms, .GlobalEnv)
        docTerms <<- newDocTerms
    }
    
    if(exists('docDist', where = .GlobalEnv)){
        n = dim(docDist)[1] + dim(newDocDist)[1]
        # JS divergences between new docs and old
        allDocDist = matrix(nrow = n, ncol = n)
        rownames(allDocDist) = append(rownames(docDist),rownames(newDocDist))
        colnames(allDocDist) = append(colnames(docDist),colnames(newDocDist))
        allDocDist[rownames(docDist),colnames(docDist)] = docDist
        allDocDist[rownames(newDocDist),colnames(newDocDist)] = newDocDist
        
        for(newDoc in rownames(newDocDist)){
            for(oldDoc in colnames(docDist)){
                #print(newDoc)
                #print(oldDoc)
                p = rep(0,length(topicTerms))
                q = rep(0,length(topicTerms))
                p[docTopics[[oldDoc]]$topic] = docTopics[[oldDoc]]$weight
                q[newDocTopics[[newDoc]]$topic] = newDocTopics[[newDoc]]$weight
                #print(p)
                #print(q)
                div = JSDivergence(p,q)/log(2)
                rowIndex = match(newDoc, rownames(allDocDist))
                colIndex = match(oldDoc, rownames(allDocDist))
                allDocDist[rowIndex, colIndex] = div
                allDocDist[colIndex,rowIndex] = div
            }
        }
        #assign('docDist', allDocDist, envir = .GlobalEnv)
        docDist <<- allDocDist
    } else {
        #assign('docDist', newDocDist, envir = .GlobalEnv)
        docDist <<- newDocDist
    }
    
    docDistMat <- as.dist(docDist)
    #assign('docClusters', hclust(docDistMat, method = 'ward.D'), envir = .GlobalEnv)
    docClusters <<- hclust(docDistMat, method = 'ward.D')
    docTopicNetworkAllLinks <<- rbindlist(docTopics,idcol = 'doc')
    setkeyv(docTopicNetworkAllLinks, c('doc','topic'))
}




###########

# Read in topic/term data.  Convert list objects to environments for hashable single-item references- scalable to larger corpora
idf = jsonlite::fromJSON('data/idf.json',simplifyDataFrame = T,simplifyVector = T)
idf <- as.environment(idf)

topicTerms = jsonlite::fromJSON('data/topicTerms.json',simplifyDataFrame = T,simplifyVector = T)
topicLabels = sapply(names(topicTerms), function(topic){
                                        paste0(topic,': ',paste0(topicTerms[[as.character(topic)]]$token[1:10], collapse = ', '))
                                        })

ingestNewDocs(docDir = 'data')

# D3 category 20b scale for labelling clusters:
nodePalette <- c('#393b79', ' #5254a3', ' #6b6ecf', ' #9c9ede', ' #637939', ' #8ca252', ' #b5cf6b', ' #cedb9c', ' #8c6d31', ' #bd9e39', 
                 ' #e7ba52', ' #e7cb94', ' #843c39', ' #ad494a', ' #d6616b', ' #e7969c', ' #7b4173', ' #a55194', ' #ce6dbd', ' #de9ed6')
# Heatmap colors for bars in topic barplots:
heatColors <- rev(heat.colors(100))
