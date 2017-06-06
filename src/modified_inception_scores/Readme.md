Code for Calculating the modified inception scores as explained in the paper.

    splits=10 			# the number of splits to average the score over
    print(preds)		# Predicted labels using the output of the transfer learnt architecture
    scores = []
    # Calculating the inception score
    for i in range(splits):
        part = preds[argmax==i]
        logp= np.log(part)
        self = np.sum(part*logp,axis=1)
        cross = np.mean(np.dot(part,np.transpose(logp)),axis=1)
        diff = self - cross
        kl = np.mean(self - cross)
        kl1 = []
        for j in range(splits):
            diffj = diff[(j * diff.shape[0] // splits):((j+ 1) * diff.shape[0] //splits)]
            kl1.append(np.exp(diffj.mean()))
        print("category: %s scores_mean = %.2f, scores_std = %.2f" % (classes[i], np.mean(kl1),np.std(kl1)))
        scores.append(np.exp(kl))
    print("scores_mean = %.2f, scores_std = %.2f" % (np.mean(scores),
                                                     np.std(scores)))

