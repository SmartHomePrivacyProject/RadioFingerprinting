# Robust Deep-learning-based Radio Fingerprinting with Fine-Tuning

## Overview 

In this project, we study radio fingerprinting. Radio fingerprinting is a way of distinguishing a tramisster from others of the same type by analyzing raw I/Q data  in wireless communications. Each tramisster has a unique finerprint due to hardware imperfections on radio-frequency circuitry. The hardware imperhections include I/Q imbalance, phase noise, frequency offset, and sampling off-set. Specifically, given I/Q data collected on a receiver side, a receiver decides which transmitter it is in radio fingerprinting. 

Recent studies have shown that deep learning can obtain high accuracy in radio fingerpinting. However, as wireless channels keep changing in the real world, one primary challenge is that the deep nueral networks for radio fingerprinting are not robust, where the training I/Q data are different from test I/Q data, especially in a cross-day scenario. We leverage fine-tuning to improve the robustness of dee-learning-based radio fingerprinting. 

## Testbed

We built a testbed with 1 receiver and 5 transmitters in order to collect I/Q data. Each receiver/transmitter is a HackRF One (with ANT500 antenna) running with GNU Radio. We leverage the open-source GNU Radio code from https://github.com/bastibl/gr-ieee802-11 to establish WiFi transmissions (IEEE 802.11 a/g) with BPSK 1/2 modulation scheme between the receiver and transmitter. We captured the I/Q data at 2.45 GHz center frequency with 2MHz bandwidth and a 2MHz sampling rate.

## Contact

If you have any questions, please feel free to contact us. 

Haipeng Li (li2hp@mail.uc.edu)

Nirnimesh Ghose (nghose@unl.edu)

Boyang Wang (boyang.wang@uc.edu)
