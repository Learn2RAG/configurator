import uvicorn
uvicorn.run(
    'learn2rag.ui:create_app',
    factory=True,
    interface='wsgi',
    host='0.0.0.0',  # TODO
    port=9000,  # TODO
)
