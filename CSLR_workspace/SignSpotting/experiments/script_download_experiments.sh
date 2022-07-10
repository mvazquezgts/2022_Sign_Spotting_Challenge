#!/bin/sh
mkdir OSLWL
mkdir MSSL

mkdir MSSL/EXPERIMENTO_MSSL_TRAIN_SET
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HwYDOuZdW8TroB_yE7RDy8VBmWnbbRim' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1HwYDOuZdW8TroB_yE7RDy8VBmWnbbRim" -O MSSL/EXPERIMENTO_MSSL_TRAIN_SET.zip && rm -rf /tmp/cookies.txt
unzip MSSL/EXPERIMENTO_MSSL_TRAIN_SET.zip -d MSSL
rm -r MSSL/EXPERIMENTO_MSSL_TRAIN_SET.zip


mkdir MSSL/EXPERIMENTO_MSSL_VAL_SET
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nJY4mXQQSmohBJ3C5rxgn3-J9rsc5y9q' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nJY4mXQQSmohBJ3C5rxgn3-J9rsc5y9q" -O MSSL/EXPERIMENTO_MSSL_VAL_SET.zip && rm -rf /tmp/cookies.txt
unzip MSSL/EXPERIMENTO_MSSL_VAL_SET.zip -d MSSL
rm -r MSSL/EXPERIMENTO_MSSL_VAL_SET.zip



mkdir OSLWL/EXPERIMENTO_OSLWL_QUERY_VAL_SET
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=149IKCWIkCvIcWKrkAn11b-rEom8NJnaH' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=149IKCWIkCvIcWKrkAn11b-rEom8NJnaH" -O OSLWL/EXPERIMENTO_OSLWL_QUERY_VAL_SET.zip && rm -rf /tmp/cookies.txt
unzip OSLWL/EXPERIMENTO_OSLWL_QUERY_VAL_SET.zip -d OSLWL
rm -r OSLWL/EXPERIMENTO_OSLWL_QUERY_VAL_SET.zip


mkdir OSLWL/EXPERIMENTO_OSLWL_VAL_SET
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1krXt3dBSDIXMAQWmKsYlxfQp9PJoC87t' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1krXt3dBSDIXMAQWmKsYlxfQp9PJoC87t" -O OSLWL/EXPERIMENTO_OSLWL_VAL_SET.zip && rm -rf /tmp/cookies.txt
unzip OSLWL/EXPERIMENTO_OSLWL_VAL_SET.zip -d OSLWL
rm -r OSLWL/EXPERIMENTO_OSLWL_VAL_SET.zip
