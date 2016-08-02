template <class T>
inline T __cudacc_attribute custom_learn(T fWeight, T fInfluence, T fInput) {
	return fWeight + (fInfluence*(fInput-fWeight) );
}

template <class T>
inline T __cudacc_attribute custom_gaussian_nhood (T dist, T sigmaT) {
	return exp(-pow(dist, 2.f)/(2.f*pow(sigmaT, 2.f)));
}

template <class T>
inline T __cudacc_attribute custom_rad_decay (T sigma0, T t, T lambda) {
	return std::floor(sigma0*exp(-t/lambda) + 0.5f);
}

template <class T>
inline T __cudacc_attribute custom_lrate_decay (T sigma0, T t, T lambda) {
	return sigma0*exp(-t/lambda);
}

template<class T> using custom_functor = ANN::DistFunction<T, custom_learn<T>, custom_gaussian_nhood<T>, custom_rad_decay<T>, custom_lrate_decay<T> >;
