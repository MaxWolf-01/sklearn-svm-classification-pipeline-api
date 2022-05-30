### sklearn-svm-classification-pipeline-api
A flask api for text-classification with sklearn pipelines.
___  
To try it out just clone the repository and run the following command: (assuming you have [docker](https://www.docker.com/) installed)
```
docker compose up --build
```  
OR  
  
Run `app.py` directly as a script.
___
You will find the classification service at `http://127.0.0.1:5000/`  
- To *initialize* and train the model:   
  - Send a post request to `/train` in the form of:
    ```
    [
        {
            "id": 69,
            "text": "example text",
            "label": 1
        }, 
        ...
    ]
    ```
- To predict:
  - Send a post request to `/predict`:
    ```
    {
        "id": 69,
        "text": "example text",
    }
    ```
  -  It will return a prediction:
      ```
      {    
          "id": 69,
          "label": 1,
          "confidence": 99
      }
      ```
- For info about the model parameters and accuracy:
  -  Send a get request to `/info`
