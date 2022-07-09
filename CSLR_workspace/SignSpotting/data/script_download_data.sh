#!/bin/sh
mkdir OSLWL
mkdir MSSL


# OSLWL_VAL_SET.zip   https://drive.google.com/file/d/1Qg9LzywfAmeizAigj59IsH4jzam4JTkp/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Qg9LzywfAmeizAigj59IsH4jzam4JTkp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Qg9LzywfAmeizAigj59IsH4jzam4JTkp" -O OSLWL/OSLWL_VAL_SET.zip && rm -rf /tmp/cookies.txt
unzip OSLWL/OSLWL_VAL_SET.zip -d OSLWL
rm -r OSLWL/OSLWL_VAL_SET.zip


# OSLWL_TEST_SET.zip 	https://drive.google.com/file/d/1ioiQxxh7rwwSRwWxOSMtC-DOdQpaOLp7/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ioiQxxh7rwwSRwWxOSMtC-DOdQpaOLp7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ioiQxxh7rwwSRwWxOSMtC-DOdQpaOLp7" -O OSLWL/OSLWL_TEST_SET.zip && rm -rf /tmp/cookies.txt
unzip OSLWL/OSLWL_TEST_SET.zip -d OSLWL
rm -r OSLWL/OSLWL_TEST_SET.zip 


# OSLWL_QUERY_VAL_SET.zip 	https://drive.google.com/file/d/13MtqlZVFog_27YetRpEM2o7oFWw_XJkH/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13MtqlZVFog_27YetRpEM2o7oFWw_XJkH' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13MtqlZVFog_27YetRpEM2o7oFWw_XJkH" -O OSLWL/OSLWL_QUERY_VAL_SET.zip && rm -rf /tmp/cookies.txt
unzip OSLWL/OSLWL_QUERY_VAL_SET.zip -d OSLWL
rm -r OSLWL/OSLWL_QUERY_VAL_SET.zip 



# OSLWL_QUERY_TEST_SET.zip 	https://drive.google.com/file/d/16fypr5r_m83WIPwtvsdFoK_iUDntmTfj/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16fypr5r_m83WIPwtvsdFoK_iUDntmTfj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16fypr5r_m83WIPwtvsdFoK_iUDntmTfj" -O OSLWL/OSLWL_QUERY_TEST_SET.zip && rm -rf /tmp/cookies.txt
unzip OSLWL/OSLWL_QUERY_TEST_SET.zip -d OSLWL
rm -r OSLWL/OSLWL_QUERY_TEST_SET.zip


# MSSL_VAL_SET.zip 	https://drive.google.com/file/d/1n8jl2taTyp11hmYnRMzFlxMG_ixwfbLX/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1n8jl2taTyp11hmYnRMzFlxMG_ixwfbLX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1n8jl2taTyp11hmYnRMzFlxMG_ixwfbLX" -O MSSL_VAL_SET.zip && rm -rf /tmp/cookies.txt
unzip MSSL/MSSL_VAL_SET.zip -d MSSL
rm -r MSSL/MSSL_VAL_SET.zip 


# MSSL_TRAIN_SET.zip 	https://drive.google.com/file/d/1iDPf7as2AiyaXQWGcfg4PECiG1-9B-lb/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1iDPf7as2AiyaXQWGcfg4PECiG1-9B-lb' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1iDPf7as2AiyaXQWGcfg4PECiG1-9B-lb" -O MSSL_TRAIN_SET.zip && rm -rf /tmp/cookies.txt
unzip MSSL/MSSL_TRAIN_SET.zip -d MSSL
rm -r MSSL/MSSL_TRAIN_SET.zip 


# MSSL_TEST_SET.zip 	https://drive.google.com/file/d/1kv-04NYKyZxnv-iciAM9OEKMwKIYfNl-/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kv-04NYKyZxnv-iciAM9OEKMwKIYfNl-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kv-04NYKyZxnv-iciAM9OEKMwKIYfNl-" -O MSSL_TEST_SET.zip && rm -rf /tmp/cookies.txt
unzip MSSL/MSSL_TEST_SET.zip -d MSSL
rm -r MSSL/MSSL_TEST_SET.zip