from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"
import plotly.express as px



def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    sample = np.random.normal(10, 1, 1000)
    sample_ug = UnivariateGaussian()
    sample_ug.fit(sample)
    output = (sample_ug.mu_, sample_ug.var_)
    print(output)
    print("\n")


    # Question 2 - Empirically showing sample mean is consistent
    dist = []
    ms = np.zeros(1000, ).astype(np.int64)
    for i in range(100):
        ms[i] = (i * 10) + 10
        x1 = sample[0:ms[i]]
        dist.append(abs(np.mean(x1) - 10))
    go.Figure([go.Scatter(x=ms, y=dist, mode='markers+lines',
                          name=r'$\widehat\mu$')],

              layout=go.Layout(
                  title=r"$\text{ Distance between estimated and true expectation}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="$\\text{ |estimated expectation- real expectation| }$",
                  height=300)).show()


    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = sample_ug.pdf(sample)
    go.Figure([go.Scatter(x=sample, y=pdfs, mode='markers',
                          marker=dict(color="black"), showlegend=False)],
              layout=go.Layout(
                  title=r"$\text{ Empirical pdf}$",
                  xaxis_title="$\\text{ value}$",
                  yaxis_title="$\\text{ pdf }$",
                  height=300)).show()



def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array(
        [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    sample_m = np.random.multivariate_normal(mu, sigma, 1000)
    sample_mg = MultivariateGaussian()
    sample_mg.fit(sample_m)
    print(sample_mg.mu_)
    print("\n")
    print(sample_mg.cov_)
    print("\n")


    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    log_likelihoods = np.zeros((200, 200))
    log_likelihood = np.zeros(200, )
    for j in range(200):
        for i in range(200):
            g = sample_mg.log_likelihood(
                np.array([f1[j], 0, f1[i],0]), sigma, sample_m)
            log_likelihood[i] = g
        log_likelihoods[j] = log_likelihood

    fig = px.imshow(log_likelihoods,
                    labels=dict(x="f1 value", y="f3 value",
                                color="log likelihood"),
                    x=f1,
                    y=f1
                    )
    fig.update_xaxes(side="top")
    fig.show()

    # Question 6 - Maximum likelihood
    t = np.unravel_index(log_likelihoods.argmax(), log_likelihoods.shape)
    print("(f1,f2)= (" + str(f1[t[0]])+","+str(f1[t[1]]) + ")")



if __name__ == '__main__':
    np.random.seed(0)
   # test_univariate_gaussian()
    test_multivariate_gaussian()
