
# from urllib.request import urlretrieve
# import os

# # *** Pytorch models download options ****
# if not os.path.isdir('./pt-models'):
#     print('Creating pt-models directory')
#     os.makedirs('./pt-models')

# # # # Download wav2vec 2.0 base
# # url_w2v = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt"
# # w2v_path = './pt-models/wav2vec_small.pt'
# # if not os.path.isfile(w2v_path):
# #     print('Downloading wav2vec 2.0')
# #     urlretrieve(url_w2v, w2v_path)
# #     print('Completed')

# # # Download no-reference natural speech SCOREQ
# # url_scoreq = 'https://www.dropbox.com/scl/fi/oqenocvm3rzdte3fdgozl/telephone.pt?rlkey=7hyuf691ot4yn9md0tuzqg9c8&st=kwl4gazl&dl=1'
# # model_path = './pt-models/scoreq_natspeech_nr.pt'
# # if not os.path.isfile(model_path):
# #     print('Downloading')
# #     print('SCOREQ | Mode: No-Reference | Data: Natural speech')
# #     urlretrieve(url_scoreq, model_path)
# #     print('Download completed')

# # Download fr/nmr-reference natural speech SCOREQ
# url_scoreq = 'https://zenodo.org/api/records/13860326/draft/files/adapt_nr_telephone.pt/content'
# model_path = './pt-models/adapt_nr_telephone.pt'
# if not os.path.isfile(model_path):
#     print('Downloading')
#     print('SCOREQ | Mode: No-Reference | Data: Natural speech')
#     urlretrieve(url_scoreq, model_path)
#     print('Download completed')

# # # Download no-reference synthetic speech SCOREQ
# # url_scoreq = 'https://www.dropbox.com/scl/fi/vzbtdf6f3uqiaz8ryax4e/synthetic.pt?rlkey=sf1l7djgxtpda7a2q0s2q1r8n&st=mhbkyj4z&dl=1'
# # model_path = './pt-models/scoreq_synthspeech_nr.pt'
# # if not os.path.isfile(model_path):
# #     print('Downloading')
# #     print('SCOREQ | Mode: No-Reference | Data: Synthetic speech')
# #     urlretrieve(url_scoreq, model_path)
# #     print('Download completed')

# # # Download fr/nmr-reference synthetic speech SCOREQ
# # url_scoreq = 'https://www.dropbox.com/scl/fi/vzbtdf6f3uqiaz8ryax4e/synthetic.pt?rlkey=sf1l7djgxtpda7a2q0s2q1r8n&st=mhbkyj4z&dl=1'
# # model_path = './pt-models/scoreq_synthspeech_ref.pt'
# # if not os.path.isfile(model_path):
# #     print('Downloading')
# #     print('SCOREQ | Mode: Fr/Nmr-Reference | Data: Synthetic speech')
# #     urlretrieve(url_scoreq, model_path)
# #     print('Download completed')