import uvicorn
uvicorn.run('learn2rag.ui:create_app', interface='wsgi', factory=True, host='0.0.0.0', port=9000)
