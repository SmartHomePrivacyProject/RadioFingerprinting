# Robust Deep-learning-based Radio Fingerprinting with Fine-Tuning

## Overview 

In this project, we study radio fingerprinting. Radio fingerprinting is a way of distinguishing a tramisster from others of the same type by analyzing raw I/Q data  in wireless communications. Each tramisster has a unique finerprint due to hardware imperfections on radio-frequency circuitry. The hardware imperhections include I/Q imbalance, phase noise, frequency offset, and sampling off-set. Specifically, given I/Q data collected on a receiver side, a receiver decides which transmitter it is in radio fingerprinting. 

Recent studies have shown that deep learning can obtain high accuracy in radio fingerpinting. However, as wireless channels keep changing in the real world, one primary challenge is that the deep nueral networks for radio fingerprinting are not robust, where the training I/Q data are different from test I/Q data, especially in a cross-day scenario. We leverage fine-tuning to improve the robustness of dee-learning-based radio fingerprinting. 

This repository contains GNU Radio source code for receivers and transmitters in our testbed and source code for deep neural networks and fine-tuning. 

The ```GNU_radio``` directory includes source code for receivers and transmitters using HackRF Ones. 

The ```rf``` directory includes source code for reading I/Q data and generating I/Q traces for a neural network. 

The ```fine-tuning``` directory includes source code for three convolutional neural networks and fine-tuning. 

The ```complex_rf``` directory includes source code for complex_value neural networks (no fine-tuning is supported at this point). 

## Testbed

We built a testbed with 1 receiver and 5 transmitters in order to collect I/Q data. Each receiver/transmitter is a HackRF One (with ANT500 antenna) running with GNU Radio. We leverage the open-source GNU Radio code from https://github.com/bastibl/gr-ieee802-11 to establish WiFi transmissions (IEEE 802.11 a/g) with BPSK 1/2 modulation scheme between the receiver and transmitter. We captured the I/Q data at 2.45 GHz center frequency with 2MHz bandwidth and a 2MHz sampling rate. 

We collected raw I/Q data before FFT, after FFT and after WiFi Frame Equalizer. We collected 3 transmissions from each transmitter. Each transmission lasts for 30 seconds.  We collected data from two different days in an indoor environment. Only I/Q data after WiFi Frame Equalizer are used in our current experiments. 

## Dataset 

Our raw I/Q dataset can be found below (**last modified: May 2021**). 

https://mailuc-my.sharepoint.com/:f:/g/personal/wang2ba_ucmail_uc_edu/EjXyRTpV0Y5Dn-OjKlxKg8gBZWyq2PIHy5OPgh3bf3g4fg?e=XavqoK

**Note:** the above link needs to be updated every 6 months due to certain settings of OneDrive. If you find the link is expired and you cannot access the data, please feel free to email us (boyang.wang@uc.edu). We will update the link as soon as we can. Thanks!

**Note:** the above dataset consists of raw I/Q data. To generate inputs for a neural network, you will need to use code in ```rf``` based on your parameters (e.g., the length of each I/Q trace, the number of I/Q traces per transmitter, etc.).  

## Contact

If you have any questions, please feel free to contact us. 

Haipeng Li (li2hp@mail.uc.edu)

Nirnimesh Ghose (nghose@unl.edu)

Boyang Wang (boyang.wang@uc.edu)
