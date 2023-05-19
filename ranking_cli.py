import argparse
import logging
import sys
import pandas as pd
import instance_level_calculate_correlations as icorr
import task_level_calculate_correlations as tcorr


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)


def perm_to_rank(systems, mean_rank):
    # return a list of lists of the systems ranked by mean_rank taking ties into account
    rank_to_systems = {}
    for i in range(len(systems)):
        if mean_rank[i] not in rank_to_systems:
            rank_to_systems[mean_rank[i]] = []
        rank_to_systems[mean_rank[i]].append(systems[i])
    return [rank_to_systems[rank] for rank in sorted(rank_to_systems.keys())]


def main():
    ####################
    # Common Arguments #
    ####################
    parser = argparse.ArgumentParser("Compute Different Aggregation")
    parser.add_argument("--df_to_rank", default="example_data_cli/xtreme_missing.csv", type=str, required=False,
                        help="dataframe to rank")
    parser.add_argument("--mode", type=str, default="task_level",
                        choices=['task_level', 'instance_level'],
                        help="Which level of aggregation is available")
    # add argument saying if the best score is the highest or the lowest
    parser.add_argument("--best_score", type=str, default="highest",
                        choices=['highest', 'lowest'],
                        help="Which level of aggregation is available")

    args = parser.parse_args()
    logging.info("Computing Different Aggregation {}".format(args.df_to_rank))
    logging.info("**** Level of information available is {}".format(args.mode))
    logging.info("**** The best score is the {} score".format(args.best_score))
    try:
        df_to_rank = pd.read_csv(args.df_to_rank)
        logging.info("Files opened Starting To Rank")
    except:
        logging.info("Error while loading the file ! Please check it !")
        sys.exit(1)

    if args.mode == 'task_level':
        if args.best_score == 'highest':
            # negative values of the values except for Model
            df_to_rank.iloc[:, 1:] = -df_to_rank.iloc[:, 1:]
            
        systems = df_to_rank.Model.values.tolist()
        mean_rank, mean_scores, systems = tcorr.mean_aggregation_task_level(df_to_rank)
        one_level_rank, one_level_borda_scores, systems = tcorr.one_levels_incomplete_aggregation_task_level(df_to_rank, return_counts=True)
        logging.info('In our paper we advise to use the 1 level ranking')
        logging.info('Example:   [7, 0, 2, ...] reads S0 is ranked 8th, S1 is ranked 1st, S2 is ranked 3rd, etc')
        if args.best_score == 'highest':
            logging.info('Results of the mean ranking {} : {}'.format(mean_rank, -mean_scores))
        elif args.best_score == 'lowest':
            logging.info('Results of the mean ranking {} : {}'.format(mean_rank, mean_scores))
        logging.info('Results of the 1 level ranking {} : {}'.format(one_level_rank, one_level_borda_scores))

        logging.info('*********** Final Results ***********')
        logging.info('Mean ranking : {}'.format(perm_to_rank(systems, mean_rank)))
        logging.info('1 level ranking : {}'.format(perm_to_rank(systems, one_level_rank)))
    elif args.mode == 'instance_level':
        if args.best_score == 'highest':
            # negative values of the values except for System and Utterance
            df_to_rank.iloc[:, 2:] = -df_to_rank.iloc[:, 2:]
        systems = df_to_rank.System.values.tolist()
        mean_rank, mean_scores, systems = icorr.mean_aggregation_instance_level(df_to_rank)
        one_level_rank, one_level_borda_scores, systems = icorr.one_levels_incomplete_aggregation(df_to_rank,return_counts=True)
        two_level_rank, two_level_borda_scores, systems = icorr.two_levels_incomplete_aggregation(df_to_rank)
        logging.info('In our paper we advise to use the 2 level ranking')
        logging.info('*********** Results ***********')
        logging.info('Example:   [7, 0, 2, ...] reads S0 is ranked 8th, S1 is ranked 1st, S2 is ranked 3rd, etc')
        if args.best_score == 'highest':
            logging.info('Results of the mean ranking {} : {}'.format(mean_rank, -mean_scores))
        elif args.best_score == 'lowest':
            logging.info('Results of the mean ranking {} : {}'.format(mean_rank, mean_scores))
        logging.info('Results of the 1 level ranking {} : {}'.format(one_level_rank, one_level_borda_scores))
        logging.info('Results of the 2 level ranking {} : {}'.format(two_level_rank, two_level_borda_scores))

        logging.info('*********** Final Results ***********')
        logging.info('Mean ranking : {}'.format(perm_to_rank(systems, mean_rank)))
        logging.info('1 level ranking : {}'.format(perm_to_rank(systems, one_level_rank)))
        logging.info('2 level ranking : {}'.format(perm_to_rank(systems, two_level_rank)))

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()


# write df to csv
# df_to_rank.to_csv('data_task/glue.csv', index=False)