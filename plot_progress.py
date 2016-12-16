import sys
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import IMP
import IMP.pmi
import IMP.pmi.output
import IMP.pmi.macros
try:
    import seaborn as sns  # make plots prettier
    sns.set_style("white")
except ImportError:
    pass


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


if len(sys.argv) < 2 or '-h' in sys.argv[1]:
    sys.exit("Usage: python plot_progress.py <stat_file.stat>")
else:
    stat_file = sys.argv[1]

stat = IMP.pmi.output.ProcessOutput(stat_file)
fields = stat.get_fields(['Total_Score', 'NOE_Sigma', 'NOE_Gamma',
                          'Dihedral_Kappa', 'CHARMM_BONDS',
                          'CHARMM_NONBONDED', 'NOELikelihood_Score',
                          'NOEPrior_Score', 'DihedralLikelihood_Score',
                          'DihedralPrior_Score', 'rmf_frame_index'])
frame = np.array(fields['rmf_frame_index'], dtype=np.double)
n_mean_points = int(frame.size / 100)
mean_frame = running_mean(frame, n_mean_points)
mean_frame = frame[int(n_mean_points/2):int(n_mean_points/2) + mean_frame.size]

score = np.array(fields['Total_Score'], dtype=np.double)
mean_score = running_mean(score, n_mean_points)
# Plot score components
fig = plt.figure()
ax1 = fig.add_subplot(411)
ax1.plot(frame, score, label='Total', color='k', alpha=.25)
ax1.plot(mean_frame, mean_score, color='k', alpha=.75)
ax1.axhline(y=score.min(), color='gray', linestyle='--', linewidth=1, alpha=.5)
ax1.legend()
for label in ax1.get_xticklabels():
    label.set_visible(False)

ax2 = fig.add_subplot(412, sharex=ax1)
charmm_score = (np.array(fields['CHARMM_BONDS'], dtype=np.double) +
                np.array(fields['CHARMM_NONBONDED'], dtype=np.double))
mean_charmm_score = running_mean(charmm_score, n_mean_points)
ax2.plot(frame, charmm_score, label='CHARMM', color='r', alpha=.25)
ax2.plot(mean_frame, mean_charmm_score, color='r', alpha=.75)
ax2.axhline(y=charmm_score.min(), color='gray', linestyle='--', linewidth=1, alpha=.5)
ax2.legend()
for label in ax2.get_xticklabels():
    label.set_visible(False)

ax3 = fig.add_subplot(413, sharex=ax1)
noe_score = (np.array(fields['NOELikelihood_Score'], dtype=np.double) +
             np.array(fields['NOEPrior_Score'], dtype=np.double))
mean_noe_score = running_mean(noe_score, n_mean_points)
ax3.plot(frame, noe_score, label='NOEs', color='g', alpha=.25)
ax3.plot(mean_frame, mean_noe_score, color='g', alpha=.75)
ax3.axhline(y=noe_score.min(), color='gray', linestyle='--', linewidth=1, alpha=.5)
ax3.legend()
for label in ax3.get_xticklabels():
    label.set_visible(False)

ax4 = fig.add_subplot(414, sharex=ax1)
dihedral_score = (np.array(fields['DihedralLikelihood_Score'], dtype=np.double) +
                  np.array(fields['DihedralPrior_Score'], dtype=np.double))
mean_dihedral_score = running_mean(dihedral_score, n_mean_points)
ax4.plot(frame, dihedral_score, label='Dihedrals', color='b', alpha=.25)
ax4.plot(mean_frame, mean_dihedral_score, color='b', alpha=.75)
ax4.axhline(y=dihedral_score.min(), color='gray', linestyle='--', linewidth=1, alpha=.5)
ax4.legend()
ax4.set_xlabel("Frame")
fig.suptitle("Restraint Scores Over Time")
plt.subplots_adjust(hspace=.1)
fig.savefig('scores.png', dpi=600)


# Plot distribution of score
mean = score.mean()
std = score.std()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(score, linewidth=0, label='Score', color='k', normed=True, bins=101)
ax.axvline(x=mean, color='r')
ax.axvline(x=mean + std, color='r', linestyle='--')
ax.axvline(x=mean - std, color='r', linestyle='--')
ax.legend()
ax4.set_xlabel("Frequency")
ax.set_title(r"Score Distribution ($\mu={0:.2f},\sigma={1:.2f}$)".format(mean,
                                                                         std))
plt.tight_layout()
fig.savefig('score_dist.png', dpi=600)


# Plot posterior distributions of nuisances
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(fields['NOE_Sigma'], linewidth=0, label='Sigma',
        bins=np.linspace(0, 2, 101), normed=True)
ax.hist(fields['NOE_Gamma'], linewidth=0, label='Gamma',
        bins=np.linspace(0, 2, 101), normed=True)
ax.legend()
ax4.set_xlabel("Frequency")
ax.set_title("Posterior Distribution of Nuisance Parameters")
plt.tight_layout()
fig.savefig('nuis_dist.png', dpi=600)
