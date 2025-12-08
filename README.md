# aiagent-platformUser --> POST /chat --> FastAPI Endpoint
         |
         v
     Redis (Store History)
         |
         v
   spaCy (Intent) + TextBlob (Sentiment)
         |
         v
     Response --> User