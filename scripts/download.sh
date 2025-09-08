mkdir ../dataset
wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
mv siftsmall.tar.gz ../dataset/
mv sift.tar.gz ../dataset/
cd ../dataset
tar -xvzf siftsmall.tar.gz
tar -xvzf sift.tar.gz