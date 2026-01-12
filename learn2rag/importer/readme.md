# Learn2RAG importer

This application is an importer for data that is to be used within the learn2rag pipeline application

Author: IFDT, KM
Version: 0.0.4

## Installation
    
- to better determine filetypes, make sure all requirements are met
- install dependencies
```
apt install libgl1 libmagic1
```
- install magic1.dll if you are using windows

- install libmagic on MacOS (with Homebrew)
```
brew install libmagic
```

## Configuration
- change /config/config.json according to your needs. Add a entry for each loader that you want to configure (see examples)
- Basic Structure for the config.json is  

```
{
    "loaders": [
        {
            "loader_type": "[TYPE_OF_LOADER]",
            [options_for the loader]
        },
        {
            "loader_type": "[TYPE_OF_LOADER]",
            [options_for the loader]
        },
        {
            ...
        }
    ]
}   
```


## Output
- output will be produced as json in the main directory in loaded_documents.json

### Example Config and Result for DirectoryLoader
DirectoryLoader will parse configured directories for files that can be mapped to a text input. 
Currently processed files are: .csv, doc, docx, .eml, epub, html, json, md, odt, .pdf, ppt, pptx, .rst, .rtf, .txt, .tsv, .cls, .xlsx, .xml

To config a directory, add to the config.json a section und "loaders" like

```
 {
   "loader_type": "DirectoryLoader",
   "path": "C:\\Users\\foo",
   "recursive": "True"
 },
```

where 
- "loader_type" is set to "DirectoryLoader" to specify the use of the DirectoryLoader
- "path" is set to the Directory in the filesystem that is to be processed
- "recursive" can be set to "True" or "False" and will specifiy whether to process all subdirectories of the given directory

All results will be written to the the loaded_documents.json for each File in the path an entry like this will be generated 

```
{
    "metadata": {
        "source": "C:C:\\Users\\foo\\Revised Manuscript_Text categorization approach.docx",
        "content_hash": "e18e509d138cf86c22df0b0dfafc5ca5b8f1e266f5e3470de68190f3ebe495b0",
        "source_path": "C:\\Users\\foo",
        "file_extension": "docx",
        "process_date": "2025-07-28",
        "process_time": "14:42:02",
        "loader_type": "DirectoryLoader"
    },
    "content": "A Corpus-based Real-time Text Classification and Tagging Approach for Social Data..."
},
```

where
- medatdata will contain metadata to the files processed
- content will hold the actual text content of the file

### Example Config and Result for HTMLoader
HTMLoader will parse configured URLs and extract text/metadata of the webpage

To config a URL, add to the config.json a section und "loaders" like

```
 {
   "loader_type": "HTMLLoader",
   "url": "https://learn2rag.de",
   "depth": 0
 },
```

where 
- "loader_type" is set to "HTMLLoader" to specify the use of the HTMLLoader
- "url" is set to the URL of where the webpage can be found
- "depth" whether to process pages that are linked on the page to the depth of x. 0 means to do not process links, 1 will process all links directly found on the page, 2 will process links found in linked pages and so on. Each page is processed only once even if linked multiple times.

All results will be written to the the loaded_documents.json for each File in the path an entry like this will be generated 

```
 {
        "metadata": {
            "source": "https://learn2rag.de",
            "content_hash": "ad31e0478b3390eb4425c5b26d41c8677f79e70b6a9c1021256c04b1db091636",
            "process_date": "2025-07-28",
            "process_time": "14:42:02",
            "loader_type": "HTMLLoader",
            "meta_properties": {
                "description": "Website",
                "og:type": "website",
                "og:locale": "de_DE",
                "og:site_name": "Learn2RAG",
                "og:title": "Learn2RAG",
                "og:url": "https://learn2rag.de//",
                "og:description": "Website",
                "og:image": "https://learn2rag.de//assets/images/Learn2RAG_Header.png",
                "viewport": "width=device-width, initial-scale=1.0"
            }
        },
        "content": "\n\nWorkshops 2025\n\nIm September und Oktober 2025 organisieren wir Workshops zum Thema RAG. Mehr dazu hier\n\nIn der heutigen schnelllebigen Geschäftswelt sind Unternehmen und öffentliche Einrichtungen gefordert, ihre Daten effizient zu nutzen, um w..."
    }
```

where
- medatdata will contain metadata to the pages processed
    - mata_properties will hold all meta_tags set in the webpage itself
- content will hold the actual text content of the page as text
  


## Changelog
- v0.0.1
  - initial testing release, allows to import files from a directory (DirectoryLoader)
- v0.0.2
  - added import of Webpages (HTMLReader)
- v0.0.3
  - added content hash for HTMLReader and config examples
- v0.0.4
  - updated dependencies
  - better ouput description for files loaded
  - fixed error if only HTMLLoader is used
 - v0.0.5
  - removed keyboard monitoring, since not running interactive mode
  - changed to relative path config
  - supress magic warning
  - added permission info for metadata in directory loader