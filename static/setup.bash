#!/bin/bash
yum update -y

yum install httpd -y
service httpd start
chkconfig httpd on

wget -O /var/www/cgi-bin/analyse.py https://lsa-stock-market.appspot.com/cacheavoid/analyse.py
chmod +x /var/www/cgi-bin/analyse.py
/var/www/cgi-bin/analyse.py
wget https://lsa-stock-market.appspot.com/cacheavoid/hello.py -P /var/www/cgi-bin
chmod +x /var/www/cgi-bin/hello.py
