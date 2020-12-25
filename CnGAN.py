import tensorflow as tf
import numpy as np
import sys
import pickle as pk
import os
import shutil
import pandas as pd
import math
import random


class moreCondsNeuFM():
    def __init__(self, args):

        self.epochs_offline_training = int(sys.argv[1])
        self.epochs_online_training = int(sys.argv[2])
        self.offline_online_threshold_interval_1_idx = int(args[3])  # Intervals higher than this value will form the test set
        self.encoded_dim = int(args[4])
        self.input_dim = int(args[5])
        self.keep_prob_val = float(args[6])
        self.lamda = float(args[7])
        self.encoded_to_generated = float(args[8])
        self.d_2_g_train_ratio = int(args[9])
        self.gan_2_r_train_ratio = int(args[10])
        self.rec_train_times = int(args[11])
        self.vid_enc_size = int(args[12])
        self.mismatch_ratio_for_d = float(args[13])
        self.gen_ratio_for_g = float(args[14])
        self.min_tw_int_count = int(args[15])  # minimum number of interactions allowed on twitter
        self.min_yt_int_count = int(args[16])  # minimum number of interactions allowed on youtube
        self.sample_users_size = int(args[17])

        self.input_yt_dim = self.input_dim
        self.input_tw_dim = self.input_dim

        self.folder_root = '../../GAN_DATA/'
        self.files_prefix = self.folder_root + 'softmax_mintw_' + str(self.min_tw_int_count) + '_minyt_' + str(self.min_yt_int_count) + '_inputdim_' + str(self.input_dim) + '_'
        self.folder_path_for_gen_results = os.getcwd() + '/Results/gen_data/'
        self.UNDERSCOPE = '_'

        self.write_file_prefix = 'file_18_Proposed_Radical' + str(self.epochs_offline_training) + self.UNDERSCOPE + str(self.epochs_online_training) + self.UNDERSCOPE + str(self.offline_online_threshold_interval_1_idx) + self.UNDERSCOPE + str(self.encoded_dim) + \
                                 self.UNDERSCOPE + str(self.input_dim) + self.UNDERSCOPE + str(self.keep_prob_val) + self.UNDERSCOPE + str(self.lamda) + self.UNDERSCOPE + str(self.encoded_to_generated) + self.UNDERSCOPE + \
                                 str(self.d_2_g_train_ratio) + self.UNDERSCOPE + str(self.gan_2_r_train_ratio) + self.UNDERSCOPE + str(self.vid_enc_size) + self.UNDERSCOPE + str(self.mismatch_ratio_for_d) + self.UNDERSCOPE + str(self.gen_ratio_for_g)

        #  load data
        self.load_data()

        self._init_graph()

    def _init_graph(self):

        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):

            self.keep_prob = tf.placeholder_with_default(1.0, shape=())
            self.X_YT = tf.placeholder(tf.float32, shape=[None, self.input_yt_dim])  # inputs
            self.Y_TW = tf.placeholder(tf.float32, shape=[None, self.input_tw_dim])  # labels
            self.Y_TW_mismatch = tf.placeholder(tf.float32, shape=[None, self.input_yt_dim])  # input Twitter distributions for the mismatching pair

            # input placeholders for pairwise loss
            self.YT_inp_1 = tf.placeholder(tf.float32, shape=[None, self.input_yt_dim])  # input YouTube distributions of the user who interacted with the video, to get the pairwise loss
            self.YT_inp_2 = tf.placeholder(tf.float32, shape=[None, self.input_yt_dim])  # input YouTube interactions of the user who did not interact with the video, to get the pairwise loss
            self.TW_real_1 = tf.placeholder(tf.float32, shape=[None, self.input_tw_dim])
            self.TW_real_2 = tf.placeholder(tf.float32, shape=[None, self.input_tw_dim])
            self.X_vid_id = tf.placeholder(tf.int32, shape=())  # video id being considered for the pairwise loss
            self.RTMINUS_inp_1 = tf.placeholder(tf.float32, shape=[None, self.total_number_of_videos])  # Input Rtminus user interaction history for the user who interacted with the video, to get the pairwise loss
            self.RTMINUS_inp_2 = tf.placeholder(tf.float32, shape=[None, self.total_number_of_videos])  # Input Rtminus user interaction history for the user who did not interact with the video, to get the pairwise loss

            # Variables.
            self.weights = self._initialize_weights()

            d_prob_real, d_logit_real = self.conditioned_discriminator(self.encoder_target(self.X_YT), self.encoder_source(self.Y_TW))
            d_prob_fake, d_logit_fake = self.conditioned_discriminator(self.encoder_target(self.X_YT), self.conditioned_generator(self.encoder_target(self.X_YT)))
            d_prob_mismatch, d_logit_mismatch = self.conditioned_discriminator(self.encoder_target(self.X_YT), self.encoder_source(self.Y_TW_mismatch))

            self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
            self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))
            self.D_loss_mismatch = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_mismatch, labels=tf.zeros_like(d_logit_mismatch)))

            self.G_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))
            # self.G_loss_content = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.conditioned_generator(self.encoder_target(self.X_YT)), labels=tf.nn.sigmoid(self.encoder_source(self.Y_TW))))
            # self.G_loss_content = tf.losses.log_loss(tf.nn.sigmoid(self.conditioned_generator(self.encoder_target(self.X_YT))), tf.nn.sigmoid(self.encoder_source(self.Y_TW)), weights=1.0, epsilon=1e-07, scope=None)
            # self.G_loss_content = tf.losses.log_loss(self.conditioned_generator(self.encoder_target(self.X_YT)), self.encoder_source(self.Y_TW), weights=1.0, epsilon=1e-07, scope=None)
            self.G_loss_content = tf.losses.absolute_difference(self.conditioned_generator(self.encoder_target(self.X_YT)), self.encoder_source(self.Y_TW), weights=1.0, scope=None)

            self.D_loss = self.D_loss_real + self.D_loss_fake + self.mismatch_ratio_for_d * self.D_loss_mismatch
            self.G_loss = self.G_loss_fake + self.gen_ratio_for_g * self.G_loss_content

            self.D_loss_vanilla = self.D_loss_real + self.D_loss_fake
            self.G_loss_vanilla = self.G_loss_fake

            _, rec_logit_usr1_real_data = self.recommender_sole(self.encoder_source(self.TW_real_1), self.encoder_target(self.YT_inp_1), self.RTMINUS_inp_1, self.X_vid_id)
            _, rec_logit_usr2_real_data = self.recommender_sole(self.encoder_source(self.TW_real_2), self.encoder_target(self.YT_inp_2), self.RTMINUS_inp_2, self.X_vid_id)

            _, rec_logit_usr1_gen_data = self.recommender_sole(self.conditioned_generator(self.encoder_target(self.YT_inp_1)), self.encoder_target(self.YT_inp_1), self.RTMINUS_inp_1, self.X_vid_id)
            _, rec_logit_usr2_gen_data = self.recommender_sole(self.conditioned_generator(self.encoder_target(self.YT_inp_2)), self.encoder_target(self.YT_inp_2), self.RTMINUS_inp_2, self.X_vid_id)

            _, self.Rec_logit_real_usr_sole = self.recommender_sole(self.encoder_source(self.TW_real_1), self.encoder_target(self.YT_inp_1), self.RTMINUS_inp_1, self.X_vid_id)
            _, self.Rec_logit_gen_usr_sole = self.recommender_sole(self.conditioned_generator(self.encoder_target(self.YT_inp_1)), self.encoder_target(self.YT_inp_1), self.RTMINUS_inp_1, self.X_vid_id)

            self.Rec_loss_real_data = tf.reduce_mean(-1 * tf.log(tf.nn.sigmoid(tf.subtract(rec_logit_usr1_real_data, rec_logit_usr2_real_data)))) + self.lamda * (
                    tf.square(tf.reduce_sum(self.weights['Rec_W1']))
                    + tf.square(tf.reduce_sum(self.weights['Rec_b1']))
            )

            self.Rec_loss_gen_data = tf.reduce_mean(-1 * tf.log(tf.nn.sigmoid(tf.subtract(rec_logit_usr1_gen_data, rec_logit_usr2_gen_data)))) + self.lamda * (
                    tf.square(tf.reduce_sum(self.weights['Rec_W1']))
                    + tf.square(tf.reduce_sum(self.weights['Rec_b1']))
            )

            self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.weights['theta_D'])
            self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.weights['theta_G'])
            self.Rec_solver_real_data = tf.train.AdamOptimizer().minimize(self.Rec_loss_real_data, var_list=self.weights['theta_Rec_n_G'])
            self.Rec_solver_gen_data = tf.train.AdamOptimizer().minimize(self.Rec_loss_gen_data, var_list=self.weights['theta_Rec_n_G'])

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def create_results_folder(self, gen_results_root_folder):
        shutil.rmtree(gen_results_root_folder)
        oldmask = os.umask(000)
        os.makedirs(gen_results_root_folder, 0o0755)
        os.umask(oldmask)

    def _initialize_weights(self):
        all_weights = dict()

        " Encoder Net Model"
        all_weights['Enc_W3_source'] = tf.Variable(self.xavier_init([self.input_tw_dim, self.encoded_dim]))
        all_weights['Enc_b3_source'] = tf.Variable(tf.zeros(shape=[self.encoded_dim]))

        all_weights['Enc_W3_target'] = tf.Variable(self.xavier_init([self.input_yt_dim, self.encoded_dim]))
        all_weights['Enc_b3_target'] = tf.Variable(tf.zeros(shape=[self.encoded_dim]))

        """ Discriminator Net model """
        all_weights['D_W1'] = tf.Variable(self.xavier_init([self.encoded_dim + self.encoded_dim, 1]))
        all_weights['D_b1'] = tf.Variable(tf.zeros(shape=[1]))
        all_weights['theta_D'] = [all_weights['D_W1'], all_weights['D_b1'], all_weights['Enc_W3_source'], all_weights['Enc_b3_source'], all_weights['Enc_W3_target'], all_weights['Enc_b3_target']]

        """ Generator Net model """
        all_weights['G_W1'] = tf.Variable(self.xavier_init([self.encoded_dim, self.encoded_dim]))
        all_weights['G_b1'] = tf.Variable(tf.zeros(shape=[self.encoded_dim]))
        all_weights['theta_G'] = [all_weights['G_W1'], all_weights['G_b1']]

        """ Recommender Net model"""
        network_phi_input_layer_size = 2 * self.encoded_dim + self.total_number_of_videos
        vid_encoded_size = self.vid_enc_size

        all_weights['Rec_W1'] = tf.Variable(self.xavier_init([network_phi_input_layer_size, vid_encoded_size]))
        all_weights['Rec_b1'] = tf.Variable(tf.zeros(shape=[vid_encoded_size]))
        all_weights['Mat_latent_video_vectors'] = tf.Variable(self.xavier_init([self.total_number_of_videos, vid_encoded_size]))

        all_weights['theta_Rec_n_G'] = [all_weights['Rec_W1'], all_weights['Rec_b1'], all_weights['Mat_latent_video_vectors'], all_weights['G_W1'], all_weights['G_b1']]

        return all_weights

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    # For YT data encoding
    def encoder_source(self, inputs_source):
        enc_output_source = tf.nn.relu(tf.matmul(inputs_source, self.weights['Enc_W3_source']) + self.weights['Enc_b3_source'])
        return enc_output_source

    # For Tw data encoding
    def encoder_target(self, inputs_target):
        enc_output_target = tf.nn.relu(tf.matmul(inputs_target, self.weights['Enc_W3_target']) + self.weights['Enc_b3_target'])
        return enc_output_target

    # discriminator without dropout
    def conditioned_discriminator(self, x, y):
        inputs = tf.concat(axis=1, values=[x, y])
        d_logit = tf.nn.relu(tf.matmul(inputs, self.weights['D_W1']) + self.weights['D_b1'])

        # d_prob = tf.nn.tanh(d_logit)
        d_prob = tf.nn.sigmoid(d_logit)
        return d_prob, d_logit

    # generator without dropout
    def conditioned_generator(self, z_yt_enc):
        gen_log_prob = tf.nn.relu(tf.matmul(z_yt_enc, self.weights['G_W1']) + self.weights['G_b1'])
        gen_prob = tf.nn.sigmoid(gen_log_prob)
        # return gen_log_prob
        return gen_prob

    # save the object to a file
    def save_obj(self, obj, file_name):
        with open(file_name, 'wb') as f:
            pk.dump(obj, f, protocol=2)

    # recommender part, considers recommendations by taking the inner product of the video and user latent factors. conducts recommendations for both interacted and non interacted users
    # @ params:
    # gen_tw_dists_int_usrs     : generated Twitter distributions of the users who have an interaction with the video <specified using 'latent_video_vector' parameter>, at the considered time interval
    # enc_yt_dists_int_usrs     : YouTube distributions of the corresponding interacted users at the considered time interval
    # gen_tw_dists_non_int_usrs : generated Twitter distributions of the users who does not have an interaction with the video <specified using 'latent_video_vector' parameter>, at the considered time interval
    # enc_yt_dists_non_int_usrs : YouTube distributions of the corresponding non-interacted users at the considered time interval
    # video_id                  : id of the video been considered
    def recommender_pairwise(self, gen_tw_dists_int_usrs, enc_yt_dists_int_usrs, rtminus_int_usrs, gen_tw_dists_non_int_usrs, enc_yt_dists_non_int_usrs, rtminus_non_int_usrs, video_id):

        prob_pred_int_usrs, pred_int_usrs = self.recommender_sole(gen_tw_dists_int_usrs, enc_yt_dists_int_usrs, rtminus_int_usrs, video_id)
        prob_pred_non_int_usrs, pred_non_int_usrs = self.recommender_sole(gen_tw_dists_non_int_usrs, enc_yt_dists_non_int_usrs, rtminus_non_int_usrs, video_id)

        return prob_pred_int_usrs, prob_pred_non_int_usrs, pred_int_usrs, pred_non_int_usrs

    # recommender part, considers recommendations by taking the inner product of the video and user latent factors. conducts recommendations for both interacted and non interacted users
    def recommender_sole(self, gen_tw_dists_usrs, enc_yt_dists_usrs, rtminus_usrs, video_id):

        concat_input_int_usr = tf.concat([tf.concat([tf.convert_to_tensor(self.encoded_to_generated * gen_tw_dists_usrs), enc_yt_dists_usrs], 1), rtminus_usrs], 1)

        rec_h1_usr_ = tf.nn.relu(tf.matmul(concat_input_int_usr, self.weights['Rec_W1']) + self.weights['Rec_b1'])
        # rec_h1_usr = tf.nn.relu(tf.matmul(rec_h1_usr_, self.weights['Rec_W2']) + self.weights['Rec_b2'])

        latent_video_vector = self.weights['Mat_latent_video_vectors'][video_id]
        pred_int_usr = tf.matmul(rec_h1_usr_, tf.transpose(tf.expand_dims(tf.convert_to_tensor(latent_video_vector), 0)))

        return tf.nn.sigmoid(pred_int_usr), pred_int_usr

    def save_data_to_csv(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    def save_data_to_csv_with_index(self, data, filename):
        df = pd.DataFrame(data, index=[0])
        df.to_csv(filename, index=False)

    def save_data_to_csv_with_index_new(self, data, filename):
        df = pd.DataFrame.from_dict(data, orient='index')
        df.to_csv(filename, index=False)

    # Offline training phase to train the GAN and R parts in a multi-task manner
    def offline_training_phase(self, gen_all_interval_user_tw_ints_matrix, gen_all_interval_user_yt_ints_matrix, gen_all_mismatching_interval_user_tw_ints_matrix, train_rec_interval_overlapped_users_map, rec_train_interval_overlapped_user_tw_ints_matrix,
                               rec_train_interval_overlapped_user_yt_ints_matrix, rec_train_interval_overlapped_user_allintvids_in_next_interval_matrix, rec_train_interval_allintvids_overlapped_user_in_next_interval, train_rec_interval_overlapped_user_rtminus_matrix,
                               train_rec_interval_non_overlapped_users_map, rec_train_interval_non_overlapped_user_tw_ints_matrix, rec_train_interval_non_overlapped_user_yt_ints_matrix, rec_train_interval_non_overlapped_user_allintvids_in_next_interval_matrix, rec_train_interval_allintvids_non_overlapped_user_in_next_interval, train_rec_interval_non_overlapped_user_rtminus_matrix,
                               test_rec_interval_users_dict, rec_test_interval_user_tw_ints_matrix, rec_test_interval_user_yt_ints_matrix, rec_test_interval_user_pretrainedintvids_in_next_interval_matrix, rec_test_interval_pretrainedintvids_user_in_next_interval_dict, test_rec_interval_user_rtminus_matrix):

        gen_all_interval_user_tw_ints_matrix_ = gen_all_interval_user_tw_ints_matrix
        gen_all_interval_user_yt_ints_matrix_ = gen_all_interval_user_yt_ints_matrix
        gen_all_mismatching_interval_user_tw_ints_matrix_ = gen_all_mismatching_interval_user_tw_ints_matrix

        rec_train_interval_overlapped_users_dict_ = train_rec_interval_overlapped_users_map
        rec_train_interval_overlapped_user_tw_ints_matrix_ = rec_train_interval_overlapped_user_tw_ints_matrix
        rec_train_interval_overlapped_user_yt_ints_dict_matrix_ = rec_train_interval_overlapped_user_yt_ints_matrix
        rec_train_interval_overlapped_user_allintvids_in_next_interval_matrix_ = rec_train_interval_overlapped_user_allintvids_in_next_interval_matrix
        rec_train_interval_allintvids_overlapped_user_in_next_interval_dict_ = rec_train_interval_allintvids_overlapped_user_in_next_interval
        train_rec_interval_overlapped_user_rtminus_matrix_ = train_rec_interval_overlapped_user_rtminus_matrix

        rec_train_interval_non_overlapped_users_dict_ = train_rec_interval_non_overlapped_users_map
        rec_train_interval_non_overlapped_user_tw_ints_matrix_ = rec_train_interval_non_overlapped_user_tw_ints_matrix
        rec_train_interval_non_overlapped_user_yt_ints_dict_matrix_ = rec_train_interval_non_overlapped_user_yt_ints_matrix
        rec_train_interval_non_overlapped_user_allintvids_in_next_interval_matrix_ = rec_train_interval_non_overlapped_user_allintvids_in_next_interval_matrix
        rec_train_interval_allintvids_non_overlapped_user_in_next_interval_dict_ = rec_train_interval_allintvids_non_overlapped_user_in_next_interval
        train_rec_interval_non_overlapped_user_rtminus_matrix_ = train_rec_interval_non_overlapped_user_rtminus_matrix

        print('OFFLINE TRAINING ...\n')

        for epoch in range(1, self.epochs_offline_training + 1):
            print('EPOCH', epoch)
            train_count_in_epoch = 0
            train_rec_overlapped_loss_count_in_epoch = 0.0
            train_overlapped_top_10_count_in_epoch = 0.0
            train_overlapped_top_20_count_in_epoch = 0.0
            train_overlapped_top_50_count_in_epoch = 0.0
            train_overlapped_top_100_count_in_epoch = 0.0

            train_rec_non_overlapped_loss_count_in_epoch = 0.0
            train_non_overlapped_top_10_count_in_epoch = 0.0
            train_non_overlapped_top_20_count_in_epoch = 0.0
            train_non_overlapped_top_50_count_in_epoch = 0.0
            train_non_overlapped_top_100_count_in_epoch = 0.0

            for interval_1_idx in range(1, model.offline_online_threshold_interval_1_idx + 1):

                print('\tInterval', interval_1_idx)
                batch_twitter_all, batch_youtube_all = gen_all_interval_user_tw_ints_matrix_[interval_1_idx], gen_all_interval_user_yt_ints_matrix_[interval_1_idx]
                batch_twitter_mismatch_all = gen_all_mismatching_interval_user_tw_ints_matrix_[interval_1_idx]

                for _ in range(0, self.gan_2_r_train_ratio):  # Number of GAN trains to R ratio
                    for _ in range(0, self.d_2_g_train_ratio):  # Number of D trains per G train
                        _, d_loss_curr, d_loss_curr_vanilla = self.sess.run([self.D_solver, self.D_loss, self.D_loss_vanilla], feed_dict={self.X_YT: batch_youtube_all, self.Y_TW: batch_twitter_all, self.Y_TW_mismatch: batch_twitter_mismatch_all, self.keep_prob: self.keep_prob_val})
                        print('D LOSS', d_loss_curr, 'D LOSS VANILLA', d_loss_curr_vanilla)
                    _, g_loss_curr, g_loss_curr_vanilla = self.sess.run([self.G_solver, self.G_loss, self.G_loss_vanilla], feed_dict={self.X_YT: batch_youtube_all, self.Y_TW: batch_twitter_all, self.keep_prob: self.keep_prob_val})
                    print('G LOSS', g_loss_curr, 'G LOSS VANILLA', g_loss_curr_vanilla)

                # First record the test results, before training (avoids first interval where no testing data are present using the following IF condition)

                if test_rec_interval_users_dict.__contains__(interval_1_idx):
                    test_rec_users_dict_in_interval = test_rec_interval_users_dict[interval_1_idx]
                    rec_test_user_tw_ints_in_interval = rec_test_interval_user_tw_ints_matrix[interval_1_idx]
                    rec_test_user_yt_ints_matrix_in_interval = rec_test_interval_user_yt_ints_matrix[interval_1_idx]
                    rec_test_user_pretrainedintvids_in_next_interval_matrix_in_interval = rec_test_interval_user_pretrainedintvids_in_next_interval_matrix[interval_1_idx]
                    rec_test_pretrainedintvids_user_in_next_interval_dict_in_interval = rec_test_interval_pretrainedintvids_user_in_next_interval_dict[interval_1_idx]
                    test_rec_user_rtminus_matrix_in_interval = test_rec_interval_user_rtminus_matrix[interval_1_idx]

                    avg_rec_loss_in_interval_test = self.calc_avg_rec_loss_test_users(test_rec_users_dict_in_interval, rec_test_pretrainedintvids_user_in_next_interval_dict_in_interval, rec_test_user_tw_ints_in_interval, rec_test_user_yt_ints_matrix_in_interval, test_rec_user_rtminus_matrix_in_interval, 1.0)

                    avg_top_10_HR_in_interval_test, avg_top_20_HR_in_interval_test, avg_top_50_HR_in_interval_test, avg_top_100_HR_in_interval_test, avg_top_10_NDCG_in_interval_test, avg_top_20_NDCG_in_interval_test, avg_top_50_NDCG_in_interval_test, avg_top_100_NDCG_in_interval_test, avg_novelty_top_10_test, avg_novelty_top_20_test, avg_novelty_top_50_test, avg_novelty_top_100_test = self.calc_top_k_test_users(test_rec_users_dict_in_interval, rec_test_user_yt_ints_matrix_in_interval, rec_test_pretrainedintvids_user_in_next_interval_dict_in_interval, rec_test_user_pretrainedintvids_in_next_interval_matrix_in_interval, test_rec_user_rtminus_matrix_in_interval, 1.0)

                    print('\n\toffline test loss : ', avg_rec_loss_in_interval_test, ', offline test top-10 HR : ', avg_top_10_HR_in_interval_test, ', offline test top-20 HR : ', avg_top_20_HR_in_interval_test, ', offline test top-50 HR : ', avg_top_50_HR_in_interval_test, ', offline test top-100 HR : ', avg_top_100_HR_in_interval_test)

                for idxx in range(0, self.rec_train_times):  # Number of GAN trains to R ratio
                    print('train X ', idxx)
                    # Recommender training per interval for overlapped users
                    train_rec_interval_overlapped_users_map = rec_train_interval_overlapped_users_dict_[interval_1_idx]
                    rec_train_allintvidsinnextinterval_overlapped_user_in_interval = rec_train_interval_allintvids_overlapped_user_in_next_interval_dict_[interval_1_idx]
                    rec_train_overlapped_user_tw_ints_in_interval = rec_train_interval_overlapped_user_tw_ints_matrix_[interval_1_idx]
                    rec_train_overlapped_user_yt_ints_in_interval = rec_train_interval_overlapped_user_yt_ints_dict_matrix_[interval_1_idx]
                    train_rec_overlapped_user_rtminus_in_interval = train_rec_interval_overlapped_user_rtminus_matrix_[interval_1_idx]
                    rec_train_overlapped_user_allintvidsinnextinterval_in_interval = rec_train_interval_overlapped_user_allintvids_in_next_interval_matrix_[interval_1_idx]

                    overlapped_avg_rec_loss_in_interval = self.calc_avg_rec_loss_overlapped_users(train_rec_interval_overlapped_users_map, rec_train_allintvidsinnextinterval_overlapped_user_in_interval, rec_train_overlapped_user_tw_ints_in_interval, rec_train_overlapped_user_yt_ints_in_interval, train_rec_overlapped_user_rtminus_in_interval, self.keep_prob_val)

                    overlapped_avg_top_10_in_interval, overlapped_avg_top_20_in_interval, overlapped_avg_top_50_in_interval, overlapped_avg_top_100_in_interval = self.calc_top_k_overlapped_users(train_rec_interval_overlapped_users_map, rec_train_overlapped_user_tw_ints_in_interval, rec_train_overlapped_user_yt_ints_in_interval,
                                                                                                                                                                                                   rec_train_overlapped_user_allintvidsinnextinterval_in_interval, rec_train_allintvidsinnextinterval_overlapped_user_in_interval, train_rec_overlapped_user_rtminus_in_interval, self.keep_prob_val)

                    print('\t\toverlapped loss : ', overlapped_avg_rec_loss_in_interval, ', overlapped top-10 : ', overlapped_avg_top_10_in_interval, ', overlapped top-20 : ', overlapped_avg_top_20_in_interval, ', overlapped top-50 : ', overlapped_avg_top_50_in_interval, ', overlapped top-100 : ', overlapped_avg_top_100_in_interval)

                    train_rec_overlapped_loss_count_in_epoch += overlapped_avg_rec_loss_in_interval
                    train_overlapped_top_10_count_in_epoch += overlapped_avg_top_10_in_interval
                    train_overlapped_top_20_count_in_epoch += overlapped_avg_top_20_in_interval
                    train_overlapped_top_50_count_in_epoch += overlapped_avg_top_50_in_interval
                    train_overlapped_top_100_count_in_epoch += overlapped_avg_top_100_in_interval
                    train_count_in_epoch += 1

                    # Recommender training per interval for non overlapped users
                    train_rec_interval_non_overlapped_users_map = rec_train_interval_non_overlapped_users_dict_[interval_1_idx]
                    rec_train_allintvidsinnextinterval_non_overlapped_user_in_interval = rec_train_interval_allintvids_non_overlapped_user_in_next_interval_dict_[interval_1_idx]
                    rec_train_non_overlapped_user_tw_ints_in_interval = rec_train_interval_non_overlapped_user_tw_ints_matrix_[interval_1_idx]
                    rec_train_non_overlapped_user_yt_ints_in_interval = rec_train_interval_non_overlapped_user_yt_ints_dict_matrix_[interval_1_idx]
                    train_rec_non_overlapped_user_rtminus_in_interval = train_rec_interval_non_overlapped_user_rtminus_matrix_[interval_1_idx]
                    rec_train_non_overlapped_user_allintvidsinnextinterval_in_interval = rec_train_interval_non_overlapped_user_allintvids_in_next_interval_matrix_[interval_1_idx]

                    non_overlapped_avg_rec_loss_in_interval = self.calc_avg_rec_loss_non_overlapped_users(train_rec_interval_non_overlapped_users_map, rec_train_allintvidsinnextinterval_non_overlapped_user_in_interval, rec_train_non_overlapped_user_tw_ints_in_interval, rec_train_non_overlapped_user_yt_ints_in_interval,
                                                                                                train_rec_non_overlapped_user_rtminus_in_interval, self.keep_prob_val)

                    non_overlapped_avg_top_10_in_interval, non_overlapped_avg_top_20_in_interval, non_overlapped_avg_top_50_in_interval, non_overlapped_avg_top_100_in_interval = self.calc_top_k_non_overlapped_users(train_rec_interval_non_overlapped_users_map, rec_train_non_overlapped_user_yt_ints_in_interval, rec_train_allintvidsinnextinterval_non_overlapped_user_in_interval, rec_train_non_overlapped_user_allintvidsinnextinterval_in_interval, train_rec_non_overlapped_user_rtminus_in_interval, self.keep_prob_val)

                    print('\t\tnon overlapped loss : ', non_overlapped_avg_rec_loss_in_interval, ', non overlapped top-10 : ', non_overlapped_avg_top_10_in_interval, ', non overlapped top-20 : ', non_overlapped_avg_top_20_in_interval, ', non overlapped top-50 : ',
                          non_overlapped_avg_top_50_in_interval, ', non overlapped top-100 : ', non_overlapped_avg_top_100_in_interval)

                    train_rec_non_overlapped_loss_count_in_epoch += non_overlapped_avg_rec_loss_in_interval
                    train_non_overlapped_top_10_count_in_epoch += non_overlapped_avg_top_10_in_interval
                    train_non_overlapped_top_20_count_in_epoch += non_overlapped_avg_top_20_in_interval
                    train_non_overlapped_top_50_count_in_epoch += non_overlapped_avg_top_50_in_interval
                    train_non_overlapped_top_100_count_in_epoch += non_overlapped_avg_top_100_in_interval

            print('\n\tOffline training for epoch:', epoch)
            print('\t\tOverlapped Loss : ', float(train_rec_overlapped_loss_count_in_epoch) / train_count_in_epoch, ', Overlapped Top-10 : ', float(train_overlapped_top_10_count_in_epoch) / train_count_in_epoch, ', Overlapped Top-20 : ', float(train_overlapped_top_20_count_in_epoch) / train_count_in_epoch, ', Overlapped Top-50 : ',
                  float(train_overlapped_top_50_count_in_epoch) / train_count_in_epoch, ', Overlapped Top-100 : ', float(train_overlapped_top_100_count_in_epoch) / train_count_in_epoch, '\n')

            print('\t\tnon Overlapped Loss : ', float(train_rec_non_overlapped_loss_count_in_epoch) / train_count_in_epoch, ', non Overlapped Top-10 : ', float(train_non_overlapped_top_10_count_in_epoch) / train_count_in_epoch, ', non Overlapped Top-20 : ',
                  float(train_non_overlapped_top_20_count_in_epoch) / train_count_in_epoch, ', non Overlapped Top-50 : ', float(train_non_overlapped_top_50_count_in_epoch) / train_count_in_epoch, ', non Overlapped Top-100 : ', float(train_non_overlapped_top_100_count_in_epoch) /
                  train_count_in_epoch, '\n')
        print('Offline trainig is completed !\n')

    # Online testing phase records the test results and re train the GAN and R in an online multi-task manner for future intervals
    def online_testing_n_training_phase(self, gen_all_interval_user_tw_ints_matrix, gen_all_interval_user_yt_ints_matrix, gen_all_mismatching_interval_user_tw_ints_matrix, gen_test_interval_user_tw_ints_matrix, gen_test_interval_user_yt_ints_matrix,
                                        gen_test_mismatching_interval_user_tw_ints_matrix, test_rec_interval_users_dict, rec_test_interval_user_tw_ints_matrix, rec_test_interval_user_yt_ints_matrix, rec_test_interval_user_pretrainedintvids_in_next_interval_matrix,
                                        rec_test_interval_pretrainedintvids_user_in_next_interval_dict, test_rec_interval_user_rtminus_matrix, train_rec_interval_overlapped_users_map, rec_train_interval_overlapped_user_tw_ints_matrix, rec_train_interval_overlapped_user_yt_ints_matrix,
                                        rec_train_interval_overlapped_user_allintvids_in_next_interval_matrix, rec_train_interval_allintvids_overlapped_user_in_next_interval, train_rec_interval_overlapped_user_rtminus_matrix, train_rec_interval_non_overlapped_users_map,
                                        rec_train_interval_non_overlapped_user_tw_ints_matrix, rec_train_interval_non_overlapped_user_yt_ints_matrix, rec_train_interval_non_overlapped_user_allintvids_in_next_interval_matrix, rec_train_interval_allintvids_non_overlapped_user_in_next_interval, train_rec_interval_non_overlapped_user_rtminus_matrix):

        # GENERATOR DATA FOR BOTH TRAINING AND TESTING
        gen_all_interval_user_tw_ints_matrix_ = gen_all_interval_user_tw_ints_matrix
        gen_all_interval_user_yt_ints_matrix_ = gen_all_interval_user_yt_ints_matrix
        gen_all_mismatching_interval_user_tw_ints_matrix_ = gen_all_mismatching_interval_user_tw_ints_matrix

        gen_test_interval_user_tw_ints_matrix_ = gen_test_interval_user_tw_ints_matrix
        gen_test_interval_user_yt_ints_matrix_ = gen_test_interval_user_yt_ints_matrix
        gen_test_mismatching_interval_user_tw_ints_matrix_ = gen_test_mismatching_interval_user_tw_ints_matrix

        # DATA FOR ONLINE TESTING
        test_rec_interval_users_dict_ = test_rec_interval_users_dict
        rec_test_interval_user_tw_ints_matrix_ = rec_test_interval_user_tw_ints_matrix
        rec_test_interval_user_yt_ints_matrix_ = rec_test_interval_user_yt_ints_matrix
        rec_test_interval_user_pretrainedintvids_in_next_interval_matrix_ = rec_test_interval_user_pretrainedintvids_in_next_interval_matrix
        rec_test_interval_pretrainedintvids_user_in_next_interval_dict_ = rec_test_interval_pretrainedintvids_user_in_next_interval_dict
        test_rec_interval_user_rtminus_matrix_ = test_rec_interval_user_rtminus_matrix

        # DATA FOR ONLINE TRAINING
        # overlapped users
        rec_train_interval_overlapped_users_dict_ = train_rec_interval_overlapped_users_map
        rec_train_interval_overlapped_user_tw_ints_matrix_ = rec_train_interval_overlapped_user_tw_ints_matrix
        rec_train_interval_overlapped_user_yt_ints_matrix_ = rec_train_interval_overlapped_user_yt_ints_matrix
        rec_train_interval_overlapped_user_allintvids_in_next_interval_matrix_ = rec_train_interval_overlapped_user_allintvids_in_next_interval_matrix
        rec_train_interval_allintvids_overlapped_user_in_next_interval_dict_ = rec_train_interval_allintvids_overlapped_user_in_next_interval
        train_rec_interval_overlapped_user_rtminus_matrix_ = train_rec_interval_overlapped_user_rtminus_matrix

        # non overlapped users
        rec_train_interval_non_overlapped_users_dict_ = train_rec_interval_non_overlapped_users_map
        rec_train_interval_non_overlapped_user_tw_ints_matrix_ = rec_train_interval_non_overlapped_user_tw_ints_matrix
        rec_train_interval_non_overlapped_user_yt_ints_matrix_ = rec_train_interval_non_overlapped_user_yt_ints_matrix
        rec_train_interval_non_overlapped_user_allintvids_in_next_interval_matrix_ = rec_train_interval_non_overlapped_user_allintvids_in_next_interval_matrix
        rec_train_interval_allintvids_non_overlapped_user_in_next_interval_dict_ = rec_train_interval_allintvids_non_overlapped_user_in_next_interval
        train_rec_interval_non_overlapped_user_rtminus_matrix_ = train_rec_interval_non_overlapped_user_rtminus_matrix

        print('\n\nONLINE TESTING AND TRAINING ... \n')
        online_test_rec_loss_count = 0.0
        online_test_top_10_HR_count = 0.0
        online_test_top_20_HR_count = 0.0
        online_test_top_50_HR_count = 0.0
        online_test_top_100_HR_count = 0.0

        online_test_top_10_NDCG_count = 0.0
        online_test_top_20_NDCG_count = 0.0
        online_test_top_50_NDCG_count = 0.0
        online_test_top_100_NDCG_count = 0.0

        online_test_top_10_novelty = 0.0
        online_test_top_20_novelty = 0.0
        online_test_top_50_novelty = 0.0
        online_test_top_100_novelty = 0.0

        online_test_count = 0

        # 1. CONDUCT RECOMMENDATIONS AND RECORD RESULTS (ONLINE TESTING)
        max_interval = np.max(list(test_rec_interval_users_dict_.keys()))
        for interval_1_idx_ in range(model.offline_online_threshold_interval_1_idx + 1, max_interval + 1):
            print('INTERVAL', interval_1_idx_)
            batch_twitter_test, batch_youtube_test = gen_test_interval_user_tw_ints_matrix_[interval_1_idx_], gen_test_interval_user_yt_ints_matrix_[interval_1_idx_]
            batch_twitter_mismatch_test = gen_test_mismatching_interval_user_tw_ints_matrix_[interval_1_idx_]

            # 1. Calculates the losses for D and G, just for recordings

            d_loss_curr = self.sess.run([self.D_loss], feed_dict={self.X_YT: batch_youtube_test, self.Y_TW: batch_twitter_test, self.Y_TW_mismatch: batch_twitter_mismatch_test, self.keep_prob: 1.0})
            print('D LOSS', d_loss_curr)
            g_loss_curr = self.sess.run([self.G_loss], feed_dict={self.X_YT: batch_youtube_test, self.Y_TW: batch_twitter_test, self.keep_prob: 1.0})
            print('G LOSS', g_loss_curr)

            # 2. Conduct Recommendations and record the results

            test_rec_usrs_in_interval = test_rec_interval_users_dict_[interval_1_idx_]
            rec_test_user_tw_ints_in_interval = rec_test_interval_user_tw_ints_matrix_[interval_1_idx_]
            rec_test_user_yt_ints_in_interval = rec_test_interval_user_yt_ints_matrix_[interval_1_idx_]
            rec_test_user_pretrainedintvids_in_interval = rec_test_interval_user_pretrainedintvids_in_next_interval_matrix_[interval_1_idx_]
            rec_test_pretrainedintvids_user_in_interval = rec_test_interval_pretrainedintvids_user_in_next_interval_dict_[interval_1_idx_]
            test_rec_user_rtminus_matrix_in_interval = test_rec_interval_user_rtminus_matrix_[interval_1_idx_]

            avg_rec_loss_in_interval_test = self.calc_avg_rec_loss_test_users(test_rec_usrs_in_interval, rec_test_pretrainedintvids_user_in_interval, rec_test_user_tw_ints_in_interval, rec_test_user_yt_ints_in_interval, test_rec_user_rtminus_matrix_in_interval, 1.0)

            avg_top_10_HR_in_interval_test, avg_top_20_HR_in_interval_test, avg_top_50_HR_in_interval_test, avg_top_100_HR_in_interval_test, avg_top_10_NDCG_in_interval_test, avg_top_20_NDCG_in_interval_test, avg_top_50_NDCG_in_interval_test, avg_top_100_NDCG_in_interval_test, \
            avg_novelty_top_10_test, avg_novelty_top_20_test, avg_novelty_top_50_test, avg_novelty_top_100_test = self.calc_top_k_test_users(test_rec_usrs_in_interval, rec_test_user_yt_ints_in_interval, rec_test_pretrainedintvids_user_in_interval, rec_test_user_pretrainedintvids_in_interval, test_rec_user_rtminus_matrix_in_interval, 1.0)

            print('\n\tonline test loss : ', avg_rec_loss_in_interval_test, ', online test top-10 HR : ', avg_top_10_HR_in_interval_test, ', online test top-20 HR : ', avg_top_20_HR_in_interval_test, ', online test top-50 HR : ', avg_top_50_HR_in_interval_test, ', online test top-100 HR : ',
                  avg_top_100_HR_in_interval_test)
            print('\n\tonline test top-10 NDCG : ', avg_top_10_NDCG_in_interval_test, ', online test top-20 NDCG : ', avg_top_20_NDCG_in_interval_test, ', online test top-50 NDCG: ', avg_top_50_NDCG_in_interval_test, ', online test top-100 NDCG : ', avg_top_100_NDCG_in_interval_test)

            print('\n\tonline test top-10 NOVELTY : ', avg_novelty_top_10_test, ', online test top-20 NOVELTY : ', avg_novelty_top_20_test, ', online test top-50 NOVELTY: ', avg_novelty_top_50_test, ', online test top-100 NOVELTY : ', avg_novelty_top_100_test)

            online_test_rec_loss_count += avg_rec_loss_in_interval_test
            online_test_top_10_HR_count += avg_top_10_HR_in_interval_test
            online_test_top_20_HR_count += avg_top_20_HR_in_interval_test
            online_test_top_50_HR_count += avg_top_50_HR_in_interval_test
            online_test_top_100_HR_count += avg_top_100_HR_in_interval_test

            online_test_top_10_NDCG_count += avg_top_10_NDCG_in_interval_test
            online_test_top_20_NDCG_count += avg_top_20_NDCG_in_interval_test
            online_test_top_50_NDCG_count += avg_top_50_NDCG_in_interval_test
            online_test_top_100_NDCG_count += avg_top_100_NDCG_in_interval_test

            online_test_top_10_novelty += avg_novelty_top_10_test
            online_test_top_20_novelty += avg_novelty_top_20_test
            online_test_top_50_novelty += avg_novelty_top_50_test
            online_test_top_100_novelty += avg_novelty_top_100_test

            online_test_count += 1

            # 2. CONDUCT ONLINE TRAINING OF THE GAN AND RECOMMENDER MODELS (ONLINE TRAINING)

            overlapped_online_train_rec_loss_count = 0
            overlapped_online_train_top_10_count = 0
            overlapped_online_train_top_20_count = 0
            overlapped_online_train_top_50_count = 0
            overlapped_online_train_top_100_count = 0
            non_overlapped_online_train_rec_loss_count = 0
            non_overlapped_online_train_top_10_count = 0
            non_overlapped_online_train_top_20_count = 0
            non_overlapped_online_train_top_50_count = 0
            non_overlapped_online_train_top_100_count = 0
            online_train_count = 0

            batch_twitter_all, batch_youtube_all = gen_all_interval_user_tw_ints_matrix_[interval_1_idx_], gen_all_interval_user_yt_ints_matrix_[interval_1_idx_]
            batch_twitter_mismatch_all = gen_all_mismatching_interval_user_tw_ints_matrix_[interval_1_idx_]

            for epoch in range(1, self.epochs_online_training + 1):
                print('\n\tEpoch', epoch)
                for _ in range(0, self.gan_2_r_train_ratio):  # Number of GAN trains to R ratio
                    for _ in range(0, self.d_2_g_train_ratio):  # Number of D trains per G train
                        _, d_loss_curr = self.sess.run([self.D_solver, self.D_loss], feed_dict={self.X_YT: batch_youtube_all, self.Y_TW: batch_twitter_all, self.Y_TW_mismatch: batch_twitter_mismatch_all, self.keep_prob: self.keep_prob_val})
                        print('D LOSS', d_loss_curr)
                    _, g_loss_curr = self.sess.run([self.G_solver, self.G_loss], feed_dict={self.X_YT: batch_youtube_all, self.Y_TW: batch_twitter_all, self.keep_prob: self.keep_prob_val})
                    print('G LOSS', g_loss_curr)

                # Recommender training per interval (for overlapped users)

                train_rec_overlapped_usrs_in_interval = rec_train_interval_overlapped_users_dict_[interval_1_idx_]
                rec_train_overlapped_user_tw_ints_in_interval = rec_train_interval_overlapped_user_tw_ints_matrix_[interval_1_idx_]
                rec_train_overlapped_user_yt_ints_in_interval = rec_train_interval_overlapped_user_yt_ints_matrix_[interval_1_idx_]
                rec_train_overlapped_user_allintvids_in_interval = rec_train_interval_overlapped_user_allintvids_in_next_interval_matrix_[interval_1_idx_]
                rec_train_allintvids_overlapped_user_in_interval = rec_train_interval_allintvids_overlapped_user_in_next_interval_dict_[interval_1_idx_]
                train_rec_overlapped_user_rtminus_matrix_in_interval = train_rec_interval_overlapped_user_rtminus_matrix_[interval_1_idx_]

                overlapped_avg_rec_loss_in_interval_train = self.calc_avg_rec_loss_overlapped_users(train_rec_overlapped_usrs_in_interval, rec_train_allintvids_overlapped_user_in_interval, rec_train_overlapped_user_tw_ints_in_interval, rec_train_overlapped_user_yt_ints_in_interval, train_rec_overlapped_user_rtminus_matrix_in_interval, self.keep_prob_val)

                overlapped_avg_top_10_in_interval_train, overlapped_avg_top_20_in_interval_train, overlapped_avg_top_50_in_interval_train, overlapped_avg_top_100_in_interval_train = self.calc_top_k_overlapped_users(train_rec_overlapped_usrs_in_interval, rec_train_overlapped_user_tw_ints_in_interval,
                                                                                                                                                                                                                       rec_train_overlapped_user_yt_ints_in_interval, rec_train_overlapped_user_allintvids_in_interval, rec_train_allintvids_overlapped_user_in_interval, train_rec_overlapped_user_rtminus_matrix_in_interval, self.keep_prob_val)

                print('\t\toverlapped loss : ', overlapped_avg_rec_loss_in_interval_train, ', overlapped top-10 : ', overlapped_avg_top_10_in_interval_train, ', overlapped top-20 : ', overlapped_avg_top_20_in_interval_train, ', overlapped top-50 : ', overlapped_avg_top_50_in_interval_train,
                      ', overlapped top-100 : ', overlapped_avg_top_100_in_interval_train)

                overlapped_online_train_rec_loss_count += overlapped_avg_rec_loss_in_interval_train
                overlapped_online_train_top_10_count += overlapped_avg_top_10_in_interval_train
                overlapped_online_train_top_20_count += overlapped_avg_top_20_in_interval_train
                overlapped_online_train_top_50_count += overlapped_avg_top_50_in_interval_train
                overlapped_online_train_top_100_count += overlapped_avg_top_100_in_interval_train
                online_train_count += 1

                # Recommender training per interval (for non overlapped users)

                train_rec_non_overlapped_usrs_in_interval = rec_train_interval_non_overlapped_users_dict_[interval_1_idx_]
                rec_train_non_overlapped_user_tw_ints_in_interval = rec_train_interval_non_overlapped_user_tw_ints_matrix_[interval_1_idx_]
                rec_train_non_overlapped_user_yt_ints_in_interval = rec_train_interval_non_overlapped_user_yt_ints_matrix_[interval_1_idx_]
                rec_train_non_overlapped_user_allintvids_in_interval = rec_train_interval_non_overlapped_user_allintvids_in_next_interval_matrix_[interval_1_idx_]
                rec_train_allintvids_non_overlapped_user_in_interval = rec_train_interval_allintvids_non_overlapped_user_in_next_interval_dict_[interval_1_idx_]
                train_rec_non_overlapped_user_rtminus_matrix_in_interval = train_rec_interval_non_overlapped_user_rtminus_matrix_[interval_1_idx_]

                non_overlapped_avg_rec_loss_in_interval_train = self.calc_avg_rec_loss_non_overlapped_users(train_rec_non_overlapped_usrs_in_interval, rec_train_allintvids_non_overlapped_user_in_interval, rec_train_non_overlapped_user_tw_ints_in_interval,
                                                                                                  rec_train_non_overlapped_user_yt_ints_in_interval, train_rec_non_overlapped_user_rtminus_matrix_in_interval, self.keep_prob_val)

                non_overlapped_avg_top_10_in_interval_train, non_overlapped_avg_top_20_in_interval_train, non_overlapped_avg_top_50_in_interval_train, non_overlapped_avg_top_100_in_interval_train = self.calc_top_k_non_overlapped_users(train_rec_non_overlapped_usrs_in_interval,
                                                                                                                                                                                                                                                         rec_train_non_overlapped_user_yt_ints_in_interval,
                                                                                                                                                                                                                                                         rec_train_allintvids_non_overlapped_user_in_interval,
                                                                                                                                                                                                                                                         rec_train_non_overlapped_user_allintvids_in_interval,
                                                                                                                                                                                                                                                         train_rec_non_overlapped_user_rtminus_matrix_in_interval, self.keep_prob_val)

                print('\t\tnon overlapped loss : ', non_overlapped_avg_rec_loss_in_interval_train, ', non overlapped top-10 : ', non_overlapped_avg_top_10_in_interval_train, ', non overlapped top-20 : ', non_overlapped_avg_top_20_in_interval_train, ', non overlapped top-50 : ',
                      non_overlapped_avg_top_50_in_interval_train, ', non overlapped top-100 : ', non_overlapped_avg_top_100_in_interval_train)

                non_overlapped_online_train_rec_loss_count += non_overlapped_avg_rec_loss_in_interval_train
                non_overlapped_online_train_top_10_count += non_overlapped_avg_top_10_in_interval_train
                non_overlapped_online_train_top_20_count += non_overlapped_avg_top_20_in_interval_train
                non_overlapped_online_train_top_50_count += non_overlapped_avg_top_50_in_interval_train
                non_overlapped_online_train_top_100_count += non_overlapped_avg_top_100_in_interval_train

            print('\n\tOnline train loss : ')
            print('\t\toverlapped Loss : ', float(overlapped_online_train_rec_loss_count) / online_train_count, ', overlapped Top-10 : ', float(overlapped_online_train_top_10_count) / online_train_count, ', overlapped Top-20 : ', float(overlapped_online_train_top_20_count) / online_train_count, ', overlapped Top-50 : ',
                  float(overlapped_online_train_top_50_count) / online_train_count, ', overlapped Top-100 : ', float(overlapped_online_train_top_100_count) / online_train_count)

            print('\t\tnon overlapped Loss : ', float(non_overlapped_online_train_rec_loss_count) / online_train_count, ', non overlapped Top-10 : ', float(non_overlapped_online_train_top_10_count) / online_train_count, ', non overlapped Top-20 : ', float(non_overlapped_online_train_top_20_count) /
                  online_train_count, ', non overlapped Top-50 : ', float(non_overlapped_online_train_top_50_count) / online_train_count, ', non overlapped Top-100 : ', float(non_overlapped_online_train_top_100_count) / online_train_count)

        print('\n\nTotal online testing results: ')
        if online_test_count > 0:
            print('Loss : ', float(online_test_rec_loss_count) / online_test_count, ', Top-10 HR : ', float(online_test_top_10_HR_count) / online_test_count, ', Top-20 HR : ', float(online_test_top_20_HR_count) / online_test_count, ', Top-50 HR : ',
                  float(online_test_top_50_HR_count) / online_test_count, ', Top-100 HR : ', float(online_test_top_100_HR_count) / online_test_count)

            print('Top-10 NDCG : ', float(online_test_top_10_NDCG_count) / online_test_count, ', Top-20 NDCG : ', float(online_test_top_20_NDCG_count) / online_test_count, ', Top-50 NDCG : ', float(online_test_top_50_NDCG_count) / online_test_count, ', Top-100 NDCG : ',
                  float(online_test_top_100_NDCG_count) / online_test_count)

            print('Top-10 NOVELTY : ', float(online_test_top_10_novelty) / online_test_count, ', Top-20 NOVELTY : ', float(online_test_top_20_novelty) / online_test_count, ', Top-50 NOVELTY : ', float(online_test_top_50_novelty) / online_test_count, ', Top-100 NOVELTY : ',
                  float(online_test_top_100_novelty) / online_test_count)
        else:
            print('No test interactions')

    def calc_avg_rec_loss_overlapped_users(self, rec_train_users_in_interval_, rec_train_allintvidsinnextinterval_user_in_interval_, rec_train_user_tw_ints_in_interval_, rec_train_user_yt_ints_in_interval_, train_rec_user_rtminus_in_interval_, keep_prob_val_):

        rec_loss_sum = 0
        rec_instance_c = 0

        # b). identify the users with and without an interaction with the video in this time interval, and compute generated distributions for them

        for video_id in rec_train_allintvidsinnextinterval_user_in_interval_:  # iterates through the videos in the time interval
            tw_dist_list_users_int_with_vid = []  # list holding the user latent vectors for users who did interact with this video at this time interval
            yt_dist_list_users_int_with_vid = []
            rtminus_list_users_int_with_vid = []
            tw_dist_list_users_not_int_with_vid = []  # list holding the user latent vectors for users who did not interact with this video at this time interval
            yt_dist_list_users_non_int_with_vid = []
            rtminus_list_users_non_int_with_vid = []
            usr_idx = 0
            for user_in_interval in rec_train_users_in_interval_:  # iterates through the list of users who has an interaction at the given time interval
                for user_in_interval_int_vid in rec_train_allintvidsinnextinterval_user_in_interval_[video_id]:  # iterate through the users who has an interaction at the given time interval with the given video
                    if user_in_interval == user_in_interval_int_vid:
                        tw_dist_list_users_int_with_vid.append(rec_train_user_tw_ints_in_interval_[usr_idx])
                        yt_dist_list_users_int_with_vid.append(rec_train_user_yt_ints_in_interval_[usr_idx])
                        rtminus_list_users_int_with_vid.append(train_rec_user_rtminus_in_interval_[usr_idx])
                    else:
                        tw_dist_list_users_not_int_with_vid.append(rec_train_user_tw_ints_in_interval_[usr_idx])
                        yt_dist_list_users_non_int_with_vid.append(rec_train_user_yt_ints_in_interval_[usr_idx])
                        rtminus_list_users_non_int_with_vid.append(train_rec_user_rtminus_in_interval_[usr_idx])

                usr_idx += 1

            # c). conduct the user-pairwise recommendation task as a batch task per interval, per video
            # latent user vectors of all users who has an interaction in this interval is identified
            # also, among them, users with and without an interaction with the video is identified and their latent vectors are collected in different lists


            # Sampling of users
            no_of_non_int_users = len(yt_dist_list_users_non_int_with_vid)
            if no_of_non_int_users > self.sample_users_size:  # Need to sample some users from this list as the number of users are too high for the processing
                sample_indices = random.sample(range(0, len(yt_dist_list_users_non_int_with_vid)), self.sample_users_size)
                tw_dist_list_users_not_int_with_vid = [tw_dist_list_users_not_int_with_vid[i] for i in sample_indices]
                yt_dist_list_users_non_int_with_vid = [yt_dist_list_users_non_int_with_vid[i] for i in sample_indices]
                rtminus_list_users_non_int_with_vid = [rtminus_list_users_non_int_with_vid[i] for i in sample_indices]

            # create the permutations of interacted and non interacted user inputs (generated inputs) to be processed as a batch
            tw_dist_list_for_int_users_ = []
            tw_dist_list_for_non_int_users_ = []

            yt_dist_list_for_int_users_ = []
            yt_dist_list_for_non_int_users_ = []

            rtminus_list_for_int_users_ = []
            rtminus_list_for_non_int_users_ = []

            for int_idx in range(0, len(yt_dist_list_users_int_with_vid)):  # iterates through latent vector codes of the users who interacted with the given video in the given time interval
                for not_int_idx in range(0, len(yt_dist_list_users_non_int_with_vid)):  # iterates through latent vector codes of the users who did not interacte with the given video in the given time interval

                    tw_dist_list_for_int_users_.append(tw_dist_list_users_int_with_vid[int_idx])
                    tw_dist_list_for_non_int_users_.append(tw_dist_list_users_not_int_with_vid[not_int_idx])

                    yt_dist_list_for_int_users_.append(yt_dist_list_users_int_with_vid[int_idx])
                    yt_dist_list_for_non_int_users_.append(yt_dist_list_users_non_int_with_vid[not_int_idx])

                    rtminus_list_for_int_users_.append(rtminus_list_users_int_with_vid[int_idx])
                    rtminus_list_for_non_int_users_.append(rtminus_list_users_non_int_with_vid[not_int_idx])

            rec_loss_curr = self.sess.run(self.Rec_loss_real_data, feed_dict={self.TW_real_1: tw_dist_list_for_int_users_, self.YT_inp_1: yt_dist_list_for_int_users_, self.RTMINUS_inp_1: rtminus_list_for_int_users_, self.TW_real_2: tw_dist_list_for_non_int_users_, self.YT_inp_2: yt_dist_list_for_non_int_users_,
                                     self.RTMINUS_inp_2: rtminus_list_for_non_int_users_, self.X_vid_id: video_id, self.keep_prob: keep_prob_val_})
            self.sess.run(self.Rec_solver_real_data, feed_dict={self.TW_real_1: tw_dist_list_for_int_users_, self.YT_inp_1: yt_dist_list_for_int_users_, self.RTMINUS_inp_1: rtminus_list_for_int_users_, self.TW_real_2: tw_dist_list_for_non_int_users_, self.YT_inp_2: yt_dist_list_for_non_int_users_,
                                     self.RTMINUS_inp_2: rtminus_list_for_non_int_users_, self.X_vid_id: video_id, self.keep_prob: keep_prob_val_})

            rec_loss_sum += rec_loss_curr
            rec_instance_c += 1
        avg_rec_loss_in_interval = float(rec_loss_sum) / rec_instance_c
        return avg_rec_loss_in_interval

    def calc_avg_rec_loss_non_overlapped_users(self, rec_usrs_in_interval_, rec_pretrainedintvids_user_in_interval_, rec_user_tw_ints_in_interval_, rec_user_yt_ints_in_interval_, rec_user_rtminus_matrix_in_interval_, keep_prob_val_):

        rec_loss_sum = 0
        rec_instance_c = 0

        # b). identify the users with and without an interaction with the video in this time interval, and compute generated distributions for them

        for video_id in rec_pretrainedintvids_user_in_interval_:  # iterates through the videos in the time interval
            yt_dist_list_users_int_with_vid = []
            rtminus_list_users_int_with_vid = []
            yt_dist_list_users_non_int_with_vid = []
            rtminus_list_users_non_int_with_vid = []
            usr_idx = 0
            for user_in_interval in rec_usrs_in_interval_:  # iterates through the list of users who has an interaction at the given time interval
                for user_in_interval_int_vid in rec_pretrainedintvids_user_in_interval_[video_id]:  # iterate through the users who has an interaction at the given time interval with the given video
                    if user_in_interval == user_in_interval_int_vid:
                        yt_dist_list_users_int_with_vid.append(rec_user_yt_ints_in_interval_[usr_idx])
                        rtminus_list_users_int_with_vid.append(rec_user_rtminus_matrix_in_interval_[usr_idx])
                    else:
                        yt_dist_list_users_non_int_with_vid.append(rec_user_yt_ints_in_interval_[usr_idx])
                        rtminus_list_users_non_int_with_vid.append(rec_user_rtminus_matrix_in_interval_[usr_idx])

                usr_idx += 1

            # c). conduct the user-pairwise recommendation task as a batch task per interval, per video
            # latent user vectors of all users who has an interaction in this interval is identified
            # also, among them, users with and without an interaction with the video is identified and their latent vectors are collected in different lists

            # Sampling of users
            no_of_non_int_users = len(yt_dist_list_users_non_int_with_vid)
            if no_of_non_int_users > self.sample_users_size:  # Need to sample some users from this list as the number of users are too high for the processing
                sample_indices = random.sample(range(0, len(yt_dist_list_users_non_int_with_vid)), self.sample_users_size)
                yt_dist_list_users_non_int_with_vid = [yt_dist_list_users_non_int_with_vid[i] for i in sample_indices]
                rtminus_list_users_non_int_with_vid = [rtminus_list_users_non_int_with_vid[i] for i in sample_indices]

            # create the permutations of interacted and non interacted user inputs (generated inputs) to be processed as a batch
            yt_dist_list_for_int_users_ = []
            yt_dist_list_for_non_int_users_ = []

            rtminus_list_for_int_users_ = []
            rtminus_list_for_non_int_users_ = []

            for int_idx in range(0, len(yt_dist_list_users_int_with_vid)):  # iterates through latent vector codes of the users who interacted with the given video in the given time interval
                for not_int_idx in range(0, len(yt_dist_list_users_non_int_with_vid)):  # iterates through latent vector codes of the users who did not interacte with the given video in the given time interval

                    yt_dist_list_for_int_users_.append(yt_dist_list_users_int_with_vid[int_idx])
                    yt_dist_list_for_non_int_users_.append(yt_dist_list_users_non_int_with_vid[not_int_idx])

                    rtminus_list_for_int_users_.append(rtminus_list_users_int_with_vid[int_idx])
                    rtminus_list_for_non_int_users_.append(rtminus_list_users_non_int_with_vid[not_int_idx])

            rec_loss_curr = self.sess.run(self.Rec_loss_gen_data, feed_dict={self.YT_inp_1: yt_dist_list_for_int_users_, self.RTMINUS_inp_1: rtminus_list_for_int_users_, self.YT_inp_2: yt_dist_list_for_non_int_users_, self.RTMINUS_inp_2: rtminus_list_for_non_int_users_, self.X_vid_id: video_id,
                                                                              self.keep_prob: keep_prob_val_})
            self.sess.run(self.Rec_solver_gen_data, feed_dict={self.YT_inp_1: yt_dist_list_for_int_users_, self.RTMINUS_inp_1: rtminus_list_for_int_users_, self.YT_inp_2: yt_dist_list_for_non_int_users_, self.RTMINUS_inp_2: rtminus_list_for_non_int_users_, self.X_vid_id: video_id,
                                                                              self.keep_prob: keep_prob_val_})
            rec_loss_sum += rec_loss_curr
            rec_instance_c += 1
        avg_rec_loss_in_interval = float(rec_loss_sum) / rec_instance_c
        return avg_rec_loss_in_interval

    def calc_avg_rec_loss_test_users(self, test_rec_usrs_in_interval_, rec_test_pretrainedintvids_user_in_interval_, rec_test_user_tw_ints_in_interval_, rec_test_user_yt_ints_in_interval_, test_rec_user_rtminus_matrix_in_interval_, keep_prob_val_):

        rec_loss_sum = 0
        rec_instance_c = 0
        # b). identify the users with and without an interaction with the video in this time interval, and compute generated distributions for them
        for video_id in rec_test_pretrainedintvids_user_in_interval_:  # iterates through the videos in the time interval
            yt_dist_list_users_int_with_vid = []
            rtminus_list_users_int_with_vid = []
            yt_dist_list_users_non_int_with_vid = []
            rtminus_list_users_non_int_with_vid = []
            usr_idx = 0
            for user_in_interval in test_rec_usrs_in_interval_:  # iterates through the list of users who has an interaction at the given time interval
                for user_in_interval_int_vid in rec_test_pretrainedintvids_user_in_interval_[video_id]:  # iterate through the users who has an interaction at the given time interval with the given video
                    if user_in_interval == user_in_interval_int_vid:
                        yt_dist_list_users_int_with_vid.append(rec_test_user_yt_ints_in_interval_[usr_idx])
                        rtminus_list_users_int_with_vid.append(test_rec_user_rtminus_matrix_in_interval_[usr_idx])
                    else:
                        yt_dist_list_users_non_int_with_vid.append(rec_test_user_yt_ints_in_interval_[usr_idx])
                        rtminus_list_users_non_int_with_vid.append(test_rec_user_rtminus_matrix_in_interval_[usr_idx])
                usr_idx += 1

            # c). conduct the user-pairwise recommendation task as a batch task per interval, per video
            # latent user vectors of all users who has an interaction in this interval is identified
            # also, among them, users with and without an interaction with the video is identified and their latent vectors are collected in different lists

            # Sampling of users
            no_of_non_int_users = len(yt_dist_list_users_non_int_with_vid)
            if no_of_non_int_users > self.sample_users_size:  # Need to sample some users from this list as the number of users are too high for the processing
                sample_indices = random.sample(range(0, len(yt_dist_list_users_non_int_with_vid)), self.sample_users_size)
                yt_dist_list_users_non_int_with_vid = [yt_dist_list_users_non_int_with_vid[i] for i in sample_indices]
                rtminus_list_users_non_int_with_vid = [rtminus_list_users_non_int_with_vid[i] for i in sample_indices]
            # create the permutations of interacted and non interacted user inputs (generated inputs) to be processed as a batch
            yt_dist_list_for_int_users_ = []
            yt_dist_list_for_non_int_users_ = []

            rtminus_list_for_int_users_ = []
            rtminus_list_for_non_int_users_ = []

            for int_idx in range(0, len(yt_dist_list_users_int_with_vid)):  # iterates through latent vector codes of the users who interacted with the given video in the given time interval
                for not_int_idx in range(0, len(yt_dist_list_users_non_int_with_vid)):  # iterates through latent vector codes of the users who did not interacte with the given video in the given time interval

                    yt_dist_list_for_int_users_.append(yt_dist_list_users_int_with_vid[int_idx])
                    yt_dist_list_for_non_int_users_.append(yt_dist_list_users_non_int_with_vid[not_int_idx])

                    rtminus_list_for_int_users_.append(rtminus_list_users_int_with_vid[int_idx])
                    rtminus_list_for_non_int_users_.append(rtminus_list_users_non_int_with_vid[not_int_idx])

            rec_loss_curr = self.sess.run(self.Rec_loss_gen_data, feed_dict={self.YT_inp_1: yt_dist_list_for_int_users_, self.RTMINUS_inp_1: rtminus_list_for_int_users_, self.YT_inp_2: yt_dist_list_for_non_int_users_, self.RTMINUS_inp_2: rtminus_list_for_non_int_users_, self.X_vid_id: video_id, self.keep_prob: keep_prob_val_})

            rec_loss_sum += rec_loss_curr
            rec_instance_c += 1
        avg_rec_loss_in_interval = float(rec_loss_sum) / rec_instance_c
        # return avg_rec_loss_in_interval, tw_dist_list_users_in_interval, generated_tw_dist_for_interval_users
        return avg_rec_loss_in_interval

    def calc_top_k_overlapped_users(self, rec_train_users_in_interval_, rec_train_user_tw_ints_in_interval_, rec_train_user_yt_ints_in_interval_, rec_train_user_allintvidsinnextinterval_in_interval_, rec_train_allintvids_overlapped_user_in_interval_, train_rec_user_rtminus_in_interval_, keep_prob_val_):

        int_vid_list = list(rec_train_allintvids_overlapped_user_in_interval_.keys())
        participating_users_ratings_for_all_vids_ = []
        for vid_id_ in int_vid_list:
            ratings_for_participating_users_per_vid_ = self.sess.run(self.Rec_logit_real_usr_sole, feed_dict={self.TW_real_1: rec_train_user_tw_ints_in_interval_, self.YT_inp_1: rec_train_user_yt_ints_in_interval_, self.RTMINUS_inp_1: train_rec_user_rtminus_in_interval_, self.X_vid_id: vid_id_, self.keep_prob: keep_prob_val_})
            participating_users_ratings_for_all_vids_.append(ratings_for_participating_users_per_vid_)

        user_video_pred_matrix = np.array(participating_users_ratings_for_all_vids_).transpose().tolist()[0]

        total_int_count = 0
        match_count_top_10 = 0
        match_count_top_20 = 0
        match_count_top_50 = 0
        match_count_top_100 = 0
        user_indx = 0
        for _ in rec_train_users_in_interval_:  # iterate through the corresponding interactions of the same user in the same interval for each input instance
            actual_int_video_list_ = rec_train_user_allintvidsinnextinterval_in_interval_[user_indx]
            top_10_video_indices_ = np.array(user_video_pred_matrix[user_indx]).argsort()[-10:][::-1]  # sorted(range(len(user_video_pred_matrix[usr])), key=lambda i: user_video_pred_matrix[usr])[-K:]
            top_20_video_indices_ = np.array(user_video_pred_matrix[user_indx]).argsort()[-20:][::-1]  # sorted(range(len(user_video_pred_matrix[usr])), key=lambda i: user_video_pred_matrix[usr])[-K:]
            top_50_video_indices_ = np.array(user_video_pred_matrix[user_indx]).argsort()[-50:][::-1]  # sorted(range(len(user_video_pred_matrix[usr])), key=lambda i: user_video_pred_matrix[usr])[-K:]
            top_100_video_indices_ = np.array(user_video_pred_matrix[user_indx]).argsort()[-100:][::-1]  # sorted(range(len(user_video_pred_matrix[usr])), key=lambda i: user_video_pred_matrix[usr])[-K:]

            top_10_video_ids_ = [int_vid_list[i] for i in top_10_video_indices_]
            top_20_video_ids_ = [int_vid_list[i] for i in top_20_video_indices_]
            top_50_video_ids_ = [int_vid_list[i] for i in top_50_video_indices_]
            top_100_video_ids_ = [int_vid_list[i] for i in top_100_video_indices_]

            total_int_count += len(actual_int_video_list_)
            match_count_top_10 += len([i for i in actual_int_video_list_ if i in top_10_video_ids_])
            match_count_top_20 += len([i for i in actual_int_video_list_ if i in top_20_video_ids_])
            match_count_top_50 += len([i for i in actual_int_video_list_ if i in top_50_video_ids_])
            match_count_top_100 += len([i for i in actual_int_video_list_ if i in top_100_video_ids_])
            user_indx += 1
        avg_top_10_accuracy = float(match_count_top_10) / total_int_count
        avg_top_20_accuracy = float(match_count_top_20) / total_int_count
        avg_top_50_accuracy = float(match_count_top_50) / total_int_count
        avg_top_100_accuracy = float(match_count_top_100) / total_int_count
        return avg_top_10_accuracy, avg_top_20_accuracy, avg_top_50_accuracy, avg_top_100_accuracy

    def calc_top_k_non_overlapped_users(self, test_rec_users_dict_in_interval_, rec_test_user_yt_ints_matrix_in_interval_, rec_test_pretrainedintvids_user_in_next_interval_dict_in_interval_, rec_test_user_pretrainedintvids_in_next_interval_matrix_in_interval_,
                              test_rec_user_rtminus_matrix_in_interval_, keep_prob_val_):

        test_rec_users_dict_in_interval = test_rec_users_dict_in_interval_
        rec_test_user_yt_ints_matrix_in_interval = rec_test_user_yt_ints_matrix_in_interval_
        rec_test_pretrainedintvids_user_in_next_interval_dict_in_interval = rec_test_pretrainedintvids_user_in_next_interval_dict_in_interval_
        rec_test_user_pretrainedintvids_in_next_interval_matrix_in_interval = rec_test_user_pretrainedintvids_in_next_interval_matrix_in_interval_
        test_rec_user_rtminus_matrix_in_interval = test_rec_user_rtminus_matrix_in_interval_

        int_vid_list = list(rec_test_pretrainedintvids_user_in_next_interval_dict_in_interval.keys())
        participating_users_ratings_for_all_vids_ = []
        for vid_id_ in int_vid_list:
            ratings_for_participating_users_per_vid_ = self.sess.run(self.Rec_logit_gen_usr_sole, feed_dict={self.YT_inp_1: rec_test_user_yt_ints_matrix_in_interval, self.RTMINUS_inp_1: test_rec_user_rtminus_matrix_in_interval, self.X_vid_id: vid_id_, self.keep_prob: keep_prob_val_})
            participating_users_ratings_for_all_vids_.append(ratings_for_participating_users_per_vid_)

        user_video_pred_matrix = np.array(participating_users_ratings_for_all_vids_).transpose().tolist()[0]

        total_int_count = 0
        match_count_top_10 = 0
        match_count_top_20 = 0
        match_count_top_50 = 0
        match_count_top_100 = 0

        user_indx = 0
        for _ in test_rec_users_dict_in_interval:  # iterate through the corresponding interactions of the same user in the same interval for each input instance
            actual_int_video_list_ = rec_test_user_pretrainedintvids_in_next_interval_matrix_in_interval[user_indx]
            top_10_video_indices_ = np.array(user_video_pred_matrix[user_indx]).argsort()[-10:][::-1]  # sorted(range(len(user_video_pred_matrix[usr])), key=lambda i: user_video_pred_matrix[usr])[-K:]
            top_20_video_indices_ = np.array(user_video_pred_matrix[user_indx]).argsort()[-20:][::-1]  # sorted(range(len(user_video_pred_matrix[usr])), key=lambda i: user_video_pred_matrix[usr])[-K:]
            top_50_video_indices_ = np.array(user_video_pred_matrix[user_indx]).argsort()[-50:][::-1]  # sorted(range(len(user_video_pred_matrix[usr])), key=lambda i: user_video_pred_matrix[usr])[-K:]
            top_100_video_indices_ = np.array(user_video_pred_matrix[user_indx]).argsort()[-100:][::-1]  # sorted(range(len(user_video_pred_matrix[usr])), key=lambda i: user_video_pred_matrix[usr])[-K:]

            top_10_video_ids_ = [int_vid_list[i] for i in top_10_video_indices_]
            top_20_video_ids_ = [int_vid_list[i] for i in top_20_video_indices_]
            top_50_video_ids_ = [int_vid_list[i] for i in top_50_video_indices_]
            top_100_video_ids_ = [int_vid_list[i] for i in top_100_video_indices_]

            total_int_count += len(actual_int_video_list_)
            match_count_top_10 += len([i for i in actual_int_video_list_ if i in top_10_video_ids_])
            match_count_top_20 += len([i for i in actual_int_video_list_ if i in top_20_video_ids_])
            match_count_top_50 += len([i for i in actual_int_video_list_ if i in top_50_video_ids_])
            match_count_top_100 += len([i for i in actual_int_video_list_ if i in top_100_video_ids_])

            user_indx += 1
        avg_top_10_HR = float(match_count_top_10) / total_int_count
        avg_top_20_HR = float(match_count_top_20) / total_int_count
        avg_top_50_HR = float(match_count_top_50) / total_int_count
        avg_top_100_HR = float(match_count_top_100) / total_int_count

        return avg_top_10_HR, avg_top_20_HR, avg_top_50_HR, avg_top_100_HR

    def calc_top_k_test_users(self, test_rec_users_dict_in_interval_, rec_test_user_yt_ints_matrix_in_interval_, rec_test_pretrainedintvids_user_in_next_interval_dict_in_interval_, rec_test_user_pretrainedintvids_in_next_interval_matrix_in_interval_, test_rec_user_rtminus_matrix_in_interval_, keep_prob_val_):

        test_rec_users_dict_in_interval = test_rec_users_dict_in_interval_
        rec_test_user_yt_ints_matrix_in_interval = rec_test_user_yt_ints_matrix_in_interval_
        rec_test_pretrainedintvids_user_in_next_interval_dict_in_interval = rec_test_pretrainedintvids_user_in_next_interval_dict_in_interval_
        rec_test_user_pretrainedintvids_in_next_interval_matrix_in_interval = rec_test_user_pretrainedintvids_in_next_interval_matrix_in_interval_
        test_rec_user_rtminus_matrix_in_interval = test_rec_user_rtminus_matrix_in_interval_

        int_vid_list = list(rec_test_pretrainedintvids_user_in_next_interval_dict_in_interval.keys())
        participating_users_ratings_for_int_vids_ = []
        for vid_id_ in int_vid_list:
            ratings_for_participating_users_per_vid_ = self.sess.run(self.Rec_logit_gen_usr_sole, feed_dict={self.YT_inp_1: rec_test_user_yt_ints_matrix_in_interval, self.RTMINUS_inp_1: test_rec_user_rtminus_matrix_in_interval, self.X_vid_id: vid_id_, self.keep_prob: keep_prob_val_})
            participating_users_ratings_for_int_vids_.append(ratings_for_participating_users_per_vid_)

        user_video_pred_matrix = np.array(participating_users_ratings_for_int_vids_).transpose().tolist()[0]

        total_int_count = 0
        match_count_top_10 = 0
        match_count_top_20 = 0
        match_count_top_50 = 0
        match_count_top_100 = 0

        per_usr_avg_ndcg_top_10 = 0.0
        per_usr_avg_ndcg_top_20 = 0.0
        per_usr_avg_ndcg_top_50 = 0.0
        per_usr_avg_ndcg_top_100 = 0.0

        count_novelty_10 = 0.0
        count_novelty_20 = 0.0
        count_novelty_50 = 0.0
        count_novelty_100 = 0.0

        user_indx = 0
        for user_id in test_rec_users_dict_in_interval:  # iterate through the corresponding interactions of the same user in the same interval for each input instance
            actual_int_video_list_ = rec_test_user_pretrainedintvids_in_next_interval_matrix_in_interval[user_indx]
            top_10_video_indices_ = np.array(user_video_pred_matrix[user_indx]).argsort()[-10:][::-1]  # sorted(range(len(user_video_pred_matrix[usr])), key=lambda i: user_video_pred_matrix[usr])[-K:]
            top_20_video_indices_ = np.array(user_video_pred_matrix[user_indx]).argsort()[-20:][::-1]  # sorted(range(len(user_video_pred_matrix[usr])), key=lambda i: user_video_pred_matrix[usr])[-K:]
            top_50_video_indices_ = np.array(user_video_pred_matrix[user_indx]).argsort()[-50:][::-1]  # sorted(range(len(user_video_pred_matrix[usr])), key=lambda i: user_video_pred_matrix[usr])[-K:]
            top_100_video_indices_ = np.array(user_video_pred_matrix[user_indx]).argsort()[-100:][::-1]  # sorted(range(len(user_video_pred_matrix[usr])), key=lambda i: user_video_pred_matrix[usr])[-K:]

            top_10_video_ids_ = [int_vid_list[i] for i in top_10_video_indices_]
            top_20_video_ids_ = [int_vid_list[i] for i in top_20_video_indices_]
            top_50_video_ids_ = [int_vid_list[i] for i in top_50_video_indices_]
            top_100_video_ids_ = [int_vid_list[i] for i in top_100_video_indices_]

            total_int_count += len(actual_int_video_list_)
            match_count_top_10 += len([i for i in actual_int_video_list_ if i in top_10_video_ids_])
            match_count_top_20 += len([i for i in actual_int_video_list_ if i in top_20_video_ids_])
            match_count_top_50 += len([i for i in actual_int_video_list_ if i in top_50_video_ids_])
            match_count_top_100 += len([i for i in actual_int_video_list_ if i in top_100_video_ids_])

            per_usr_avg_ndcg_top_10 += self.get_avg_NDCG_per_list(actual_int_video_list_, top_10_video_ids_)
            per_usr_avg_ndcg_top_20 += self.get_avg_NDCG_per_list(actual_int_video_list_, top_20_video_ids_)
            per_usr_avg_ndcg_top_50 += self.get_avg_NDCG_per_list(actual_int_video_list_, top_50_video_ids_)
            per_usr_avg_ndcg_top_100 += self.get_avg_NDCG_per_list(actual_int_video_list_, top_100_video_ids_)

            novelty_10, novelty_20, novelty_50, novelty_100 = self.calc_novelty(user_id, actual_int_video_list_, test_rec_user_rtminus_matrix_in_interval_[user_indx], top_10_video_ids_, top_20_video_ids_, top_50_video_ids_,
                                                                                top_100_video_ids_)

            count_novelty_10 += novelty_10
            count_novelty_20 += novelty_20
            count_novelty_50 += novelty_50
            count_novelty_100 += novelty_100

            user_indx += 1
        avg_top_10_HR = float(match_count_top_10) / total_int_count
        avg_top_20_HR = float(match_count_top_20) / total_int_count
        avg_top_50_HR = float(match_count_top_50) / total_int_count
        avg_top_100_HR = float(match_count_top_100) / total_int_count

        avg_top_10_NDCG = float(per_usr_avg_ndcg_top_10) / user_indx
        avg_top_20_NDCG = float(per_usr_avg_ndcg_top_20) / user_indx
        avg_top_50_NDCG = float(per_usr_avg_ndcg_top_50) / user_indx
        avg_top_100_NDCG = float(per_usr_avg_ndcg_top_100) / user_indx

        avg_novelty_10 = float(count_novelty_10) / user_indx
        avg_novelty_20 = float(count_novelty_20) / user_indx
        avg_novelty_50 = float(count_novelty_50) / user_indx
        avg_novelty_100 = float(count_novelty_100) / user_indx

        return avg_top_10_HR, avg_top_20_HR, avg_top_50_HR, avg_top_100_HR, avg_top_10_NDCG, avg_top_20_NDCG, avg_top_50_NDCG, avg_top_100_NDCG, avg_novelty_10, avg_novelty_20, avg_novelty_50, avg_novelty_100

    def get_avg_NDCG_per_list(self, actual_int_item_lst, ranklist):
        count_ndcg_val = 0.0
        for item in actual_int_item_lst:
            count_ndcg_val += self.get_NDCG_per_int_item(ranklist, item)
        return float(count_ndcg_val) / len(actual_int_item_lst)

    def get_NDCG_per_int_item(self, ranklist, int_item):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == int_item:
                return math.log(2) / math.log(i + 2)
        return 0

    def get_interval_user_list_for_users_with_history(self, train_interval_user_one_hot_matrx_with_history_):
        interval_user_ids_matrix_with_history = []
        for user_one_hot_list_in_interval in train_interval_user_one_hot_matrx_with_history_:
            user_ids_in_interval_list = []
            for one_hot_encoded_usr in user_one_hot_list_in_interval:
                user_id = np.nonzero(one_hot_encoded_usr)[0][0]
                user_ids_in_interval_list.append(user_id)
            interval_user_ids_matrix_with_history.append(user_ids_in_interval_list)
        return interval_user_ids_matrix_with_history

    def calc_novelty(self, user_id, usr_actual_int_video_list_, test_rec_user_rtminus_matrix_in_interval_, top_10_video_ids_, top_20_video_ids_, top_50_video_ids_, top_100_video_ids_):

        user_id_indx_map = self.usr_id_indx_map

        user_indx = user_id_indx_map[user_id]

        Tr_l = set(np.nonzero(test_rec_user_rtminus_matrix_in_interval_)[0])
        Te_l = set(usr_actual_int_video_list_)

        R_l_10 = set(top_10_video_ids_)
        R_l_20 = set(top_20_video_ids_)
        R_l_50 = set(top_50_video_ids_)
        R_l_100 = set(top_100_video_ids_)

        novelty_10 = len(R_l_10.intersection(Te_l)) / len(Te_l) - len(R_l_10.intersection(Tr_l)) / len(Tr_l)
        novelty_20 = len(R_l_20.intersection(Te_l)) / len(Te_l) - len(R_l_20.intersection(Tr_l)) / len(Tr_l)
        novelty_50 = len(R_l_50.intersection(Te_l)) / len(Te_l) - len(R_l_50.intersection(Tr_l)) / len(Tr_l)
        novelty_100 = len(R_l_100.intersection(Te_l)) / len(Te_l) - len(R_l_100.intersection(Tr_l)) / len(Tr_l)

        return novelty_10, novelty_20, novelty_50, novelty_100

    def get_prev_int_vid_list(self, user_indx, usr_ids_matrix_with_history_in_interval_, user_vid_ids_matrx_with_history_in_interval_):
        # get user's previously interacted items
        for indx in range(0, len(usr_ids_matrix_with_history_in_interval_)):
            if user_indx == usr_ids_matrix_with_history_in_interval_[indx]:
                return user_vid_ids_matrx_with_history_in_interval_[indx]

    # load the object from the file
    def load_obj(self, file_name):
        with open(file_name, 'rb') as f:
            return pk.load(f)

    # load the object from the file
    def load_data(self):
        # save user id index list
        self.filename_user_id_indx_map = self.files_prefix + 'user_id_indx_map.pkl'
        self.filename_video_id_list = self.files_prefix + 'vid_id_list.pkl'

        self.filename_gen_all_interval_n_user_dict = self.files_prefix + 'gen_all_interval_n_user_dict.pkl'
        self.filename_gen_all_interval_n_video_dict = self.files_prefix + 'gen_all_interval_n_video_dict.pkl'
        self.filename_gen_all_interval_user_tw_ints_matrix = self.files_prefix + 'gen_all_interval_user_tw_ints_matrix.pkl'
        self.filename_gen_all_interval_user_yt_ints_matrix = self.files_prefix + 'gen_all_interval_user_yt_ints_matrix.pkl'
        self.filename_gen_all_mismatching_interval_user_tw_ints_matrix = self.files_prefix + 'gen_all_mismatching_interval_user_tw_ints_matrix.pkl'
        self.filename_gen_all_mismatching_interval_user_yt_ints_matrix = self.files_prefix + 'gen_all_mismatching_interval_user_yt_ints_matrix.pkl'
        self.filename_gen_test_interval_n_user_dict = self.files_prefix + 'gen_test_interval_n_user_dict.pkl'
        self.filename_gen_test_interval_user_tw_ints_matrix = self.files_prefix + 'gen_test_interval_user_tw_ints_matrix.pkl'
        self.filename_gen_test_interval_user_yt_ints_matrix = self.files_prefix + 'gen_test_interval_user_yt_ints_matrix.pkl'
        self.filename_gen_test_mismatching_interval_user_tw_ints_matrix = self.files_prefix + 'gen_test_mismatching_interval_user_tw_ints_matrix.pkl'
        self.filename_gen_test_mismatching_interval_user_yt_ints_matrix = self.files_prefix + 'gen_test_mismatching_interval_user_yt_ints_matrix.pkl'

        self.filename_train_rec_interval_overlapped_users_map = self.files_prefix + 'train_rec_interval_overlapped_users_map.pkl'
        self.filename_rec_train_interval_overlapped_user_tw_ints_matrix = self.files_prefix + 'rec_train_interval_overlapped_user_tw_ints_matrix.pkl'
        self.filename_rec_train_interval_overlapped_user_yt_ints_matrix = self.files_prefix + 'rec_train_interval_overlapped_user_yt_ints_matrix.pkl'
        self.filename_rec_train_interval_overlapped_user_allintvids_in_next_interval_matrix = self.files_prefix + 'rec_train_interval_overlapped_user_allintvids_in_next_interval_matrix.pkl'
        self.filename_rec_train_interval_allintvids_overlapped_user_in_next_interval = self.files_prefix + 'rec_train_interval_allintvids_overlapped_user_in_next_interval.pkl'
        self.filename_train_rec_interval_overlapped_user_rtminus_matrix = self.files_prefix + 'train_rec_interval_overlapped_user_rtminus_matrix.pkl'

        self.filename_train_rec_interval_non_overlapped_users_map = self.files_prefix + 'train_rec_interval_non_overlapped_users_map.pkl'
        self.filename_rec_train_interval_non_overlapped_user_tw_ints_matrix = self.files_prefix + 'rec_train_interval_non_overlapped_user_tw_ints_matrix.pkl'
        self.filename_rec_train_interval_non_overlapped_user_yt_ints_matrix = self.files_prefix + 'rec_train_interval_non_overlapped_user_yt_ints_matrix.pkl'
        self.filename_rec_train_interval_non_overlapped_user_allintvids_in_next_interval_matrix = self.files_prefix + 'rec_train_interval_non_overlapped_user_allintvids_in_next_interval_matrix.pkl'
        self.filename_rec_train_interval_allintvids_non_overlapped_user_in_next_interval = self.files_prefix + 'rec_train_interval_allintvids_non_overlapped_user_in_next_interval.pkl'
        self.filename_train_rec_interval_non_overlapped_user_rtminus_matrix = self.files_prefix + 'train_rec_interval_non_overlapped_user_rtminus_matrix.pkl'

        self.filename_test_rec_interval_users_dict = self.files_prefix + 'test_rec_interval_users_dict.pkl'
        self.filename_rec_test_interval_user_tw_ints_matrix = self.files_prefix + 'rec_test_interval_user_tw_ints_matrix.pkl'
        self.filename_rec_test_interval_user_yt_ints_matrix = self.files_prefix + 'rec_test_interval_user_yt_ints_matrix.pkl'
        self.filename_rec_test_interval_user_pretrainedintvids_in_next_interval_matrix = self.files_prefix + 'rec_test_interval_user_pretrainedintvids_in_next_interval_matrix.pkl'
        self.filename_rec_test_interval_pretrainedintvids_user_in_next_interval_dict = self.files_prefix + 'rec_test_interval_pretrainedintvids_user_in_next_interval_dict.pkl'
        self.filename_test_rec_interval_user_rtminus_matrix = self.files_prefix + 'test_rec_interval_user_rtminus_matrix.pkl'

        ########################################

        self.usr_id_indx_map = self.load_obj(self.filename_user_id_indx_map)
        self.video_id_list = self.load_obj(self.filename_video_id_list)

        self.gen_all_interval_n_user_dict = self.load_obj(self.filename_gen_all_interval_n_user_dict)
        self.gen_all_interval_n_video_dict = self.load_obj(self.filename_gen_all_interval_n_video_dict)
        self.gen_all_interval_user_tw_ints_matrix = self.load_obj(self.filename_gen_all_interval_user_tw_ints_matrix)
        self.gen_all_interval_user_yt_ints_matrix = self.load_obj(self.filename_gen_all_interval_user_yt_ints_matrix)
        self.gen_all_mismatching_interval_user_tw_ints_matrix = self.load_obj(self.filename_gen_all_mismatching_interval_user_tw_ints_matrix)
        self.gen_all_mismatching_interval_user_yt_ints_matrix = self.load_obj(self.filename_gen_all_mismatching_interval_user_yt_ints_matrix)
        self.gen_test_interval_n_user_dict = self.load_obj(self.filename_gen_test_interval_n_user_dict)
        self.gen_test_interval_user_tw_ints_matrix = self.load_obj(self.filename_gen_test_interval_user_tw_ints_matrix)
        self.gen_test_interval_user_yt_ints_matrix = self.load_obj(self.filename_gen_test_interval_user_yt_ints_matrix)
        self.gen_test_mismatching_interval_user_tw_ints_matrix = self.load_obj(self.filename_gen_test_mismatching_interval_user_tw_ints_matrix)
        self.gen_test_mismatching_interval_user_yt_ints_matrix = self.load_obj(self.filename_gen_test_mismatching_interval_user_yt_ints_matrix)

        self.train_rec_interval_overlapped_users_map = self.load_obj(self.filename_train_rec_interval_overlapped_users_map)
        self.rec_train_interval_overlapped_user_tw_ints_matrix = self.load_obj(self.filename_rec_train_interval_overlapped_user_tw_ints_matrix)
        self.rec_train_interval_overlapped_user_yt_ints_matrix = self.load_obj(self.filename_rec_train_interval_overlapped_user_yt_ints_matrix)
        self.rec_train_interval_overlapped_user_allintvids_in_next_interval_matrix = self.load_obj(self.filename_rec_train_interval_overlapped_user_allintvids_in_next_interval_matrix)
        self.rec_train_interval_allintvids_overlapped_user_in_next_interval = self.load_obj(self.filename_rec_train_interval_allintvids_overlapped_user_in_next_interval)
        self.train_rec_interval_overlapped_user_rtminus_matrix = self.load_obj(self.filename_train_rec_interval_overlapped_user_rtminus_matrix)

        self.train_rec_interval_non_overlapped_users_map = self.load_obj(self.filename_train_rec_interval_non_overlapped_users_map)
        self.rec_train_interval_non_overlapped_user_tw_ints_matrix = self.load_obj(self.filename_rec_train_interval_non_overlapped_user_tw_ints_matrix)
        self.rec_train_interval_non_overlapped_user_yt_ints_matrix = self.load_obj(self.filename_rec_train_interval_non_overlapped_user_yt_ints_matrix)
        self.rec_train_interval_non_overlapped_user_allintvids_in_next_interval_matrix = self.load_obj(self.filename_rec_train_interval_non_overlapped_user_allintvids_in_next_interval_matrix)
        self.rec_train_interval_allintvids_non_overlapped_user_in_next_interval = self.load_obj(self.filename_rec_train_interval_allintvids_non_overlapped_user_in_next_interval)
        self.train_rec_interval_non_overlapped_user_rtminus_matrix = self.load_obj(self.filename_train_rec_interval_non_overlapped_user_rtminus_matrix)

        self.test_rec_interval_users_dict = self.load_obj(self.filename_test_rec_interval_users_dict)
        self.rec_test_interval_user_tw_ints_matrix = self.load_obj(self.filename_rec_test_interval_user_tw_ints_matrix)
        self.rec_test_interval_user_yt_ints_matrix = self.load_obj(self.filename_rec_test_interval_user_yt_ints_matrix)
        self.rec_test_interval_user_pretrainedintvids_in_next_interval_matrix = self.load_obj(self.filename_rec_test_interval_user_pretrainedintvids_in_next_interval_matrix)
        self.rec_test_interval_pretrainedintvids_user_in_next_interval_dict = self.load_obj(self.filename_rec_test_interval_pretrainedintvids_user_in_next_interval_dict)
        self.test_rec_interval_user_rtminus_matrix = self.load_obj(self.filename_test_rec_interval_user_rtminus_matrix)

        self.total_number_of_videos = len(self.video_id_list)


if __name__ == '__main__':
    # Training
    model = moreCondsNeuFM(sys.argv)

    model.offline_training_phase(model.gen_all_interval_user_tw_ints_matrix, model.gen_all_interval_user_yt_ints_matrix, model.gen_all_mismatching_interval_user_tw_ints_matrix, model.train_rec_interval_overlapped_users_map, model.rec_train_interval_overlapped_user_tw_ints_matrix,
                                 model.rec_train_interval_overlapped_user_yt_ints_matrix, model.rec_train_interval_overlapped_user_allintvids_in_next_interval_matrix, model.rec_train_interval_allintvids_overlapped_user_in_next_interval, model.train_rec_interval_overlapped_user_rtminus_matrix,
                                 model.train_rec_interval_non_overlapped_users_map, model.rec_train_interval_non_overlapped_user_tw_ints_matrix, model.rec_train_interval_non_overlapped_user_yt_ints_matrix, model.rec_train_interval_non_overlapped_user_allintvids_in_next_interval_matrix,
                                 model.rec_train_interval_allintvids_non_overlapped_user_in_next_interval, model.train_rec_interval_non_overlapped_user_rtminus_matrix, model.test_rec_interval_users_dict, model.rec_test_interval_user_tw_ints_matrix, model.rec_test_interval_user_yt_ints_matrix, model.rec_test_interval_user_pretrainedintvids_in_next_interval_matrix, model.rec_test_interval_pretrainedintvids_user_in_next_interval_dict, model.test_rec_interval_user_rtminus_matrix)

    model.online_testing_n_training_phase(model.gen_all_interval_user_tw_ints_matrix, model.gen_all_interval_user_yt_ints_matrix, model.gen_all_mismatching_interval_user_tw_ints_matrix,
                                          model.gen_test_interval_user_tw_ints_matrix, model.gen_test_interval_user_yt_ints_matrix, model.gen_test_mismatching_interval_user_tw_ints_matrix,
                                          model.test_rec_interval_users_dict, model.rec_test_interval_user_tw_ints_matrix, model.rec_test_interval_user_yt_ints_matrix, model.rec_test_interval_user_pretrainedintvids_in_next_interval_matrix, model.rec_test_interval_pretrainedintvids_user_in_next_interval_dict,
                                          model.test_rec_interval_user_rtminus_matrix, model.train_rec_interval_overlapped_users_map, model.rec_train_interval_overlapped_user_tw_ints_matrix, model.rec_train_interval_overlapped_user_yt_ints_matrix, model.rec_train_interval_overlapped_user_allintvids_in_next_interval_matrix,
                                          model.rec_train_interval_allintvids_overlapped_user_in_next_interval,  model.train_rec_interval_overlapped_user_rtminus_matrix, model.train_rec_interval_non_overlapped_users_map, model.rec_train_interval_non_overlapped_user_tw_ints_matrix,
                                          model.rec_train_interval_non_overlapped_user_yt_ints_matrix, model.rec_train_interval_non_overlapped_user_allintvids_in_next_interval_matrix, model.rec_train_interval_allintvids_non_overlapped_user_in_next_interval, model.train_rec_interval_non_overlapped_user_rtminus_matrix)


