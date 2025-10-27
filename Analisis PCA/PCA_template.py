import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class PCA_Analysis():
    def __init__(self, data):
        self.data = data

    def PCA_2D(self):
        pca = PCA(n_components=2)
        # The wavelength columns are from index 3 to the end
        wavelength_data = self.data.iloc[:, 3:]

        # Fit AND transform the data
        pca_components = pca.fit_transform(wavelength_data)

        # Check explained variance
        explained_variance = pca.explained_variance_ratio_
        print(f"Explained variance by PC1: {explained_variance[0]:.2%}")
        print(f"Explained variance by PC2: {explained_variance[1]:.2%}")
        print(f"Total explained variance: {explained_variance.sum():.2%}")

        # Create a DataFrame with PCA results
        pca_df = pd.DataFrame(
            data=pca_components,
            columns=['PC1', 'PC2']
        )

        # Add categorical columns for plotting
        pca_df['Tratamiento'] = self.data['Tratamiento'].values
        pca_df['Planta'] = self.data['Planta'].values

        # Reorder columns to put Tratamiento and Planta first
        pca_df = pca_df[['Tratamiento', 'Planta', 'PC1', 'PC2']]

        # Direct PCA visualization approach (without using template)
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot each treatment as a separate group
        for treatment in pca_df['Tratamiento'].unique():
            subset = pca_df[pca_df['Tratamiento'] == treatment]
            ax.scatter(subset['PC1'], subset['PC2'], 
                        label=treatment, alpha=0.7, s=60)

        # Add proper labels that reflect PCA meaning
        ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]:.4} variance)')
        ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]:.4} variance)')
        ax.set_title('PCA: Treatment Groups in PC Space')
        ax.legend(title='Treatment')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show(block=True)
        
        return pca_df

    def PCA_3D(self):
        pca = PCA(n_components= 3)

        # The wavelength columns are from index 3 to the end
        wavelength_data = self.data.iloc[:, 3:]

        # Fit AND transform the data
        pca_components_3d = pca.fit_transform(wavelength_data)

        # Check explained variance
        explained_variance = pca.explained_variance_ratio_
        print(f"Explained variance by PC1: {explained_variance[0]:.2%}")
        print(f"Explained variance by PC2: {explained_variance[1]:.2%}")
        print(f"Explained variance by PC3: {explained_variance[2]:.2%}")
        print(f"Total explained variance: {explained_variance.sum():.2%}")

        # Create a DataFrame with PCA results
        pca_df = pd.DataFrame(
            data=pca_components_3d,
            columns=['PC1', 'PC2', 'PC3']
        )

        # Add categorical columns for plotting
        pca_df['Tratamiento'] = self.data['Tratamiento'].values
        pca_df['Planta'] = self.data['Planta'].values

        # Reorder columns to put Tratamiento and Planta first
        pca_df = pca_df[['Tratamiento', 'Planta', 'PC1', 'PC2', 'PC3']]


        # Create the figure and 3D axes
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Get unique treatments for color mapping
        treatments = pca_df['Tratamiento'].unique()

        # First plot all non-Fusarium treatments
        for treatment in treatments:
            subset = pca_df[pca_df['Tratamiento'] == treatment]
            ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'],
                        label=treatment, alpha=0.6, s=60, edgecolor='w')


        # Set axis labels with explained variance
        ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%} variance)', fontsize=12)
        ax.set_zlabel(f'PC3 ({explained_variance[2]:.1%} variance)', fontsize=12)

        # Set title and legend
        ax.set_title('3D PCA: Treatment Groups with Highlighted Fusarium Outlier', fontsize=14)
        ax.legend(title='Treatment', loc='upper right', bbox_to_anchor=(1.1, 1))

        # Improve perspective
        ax.view_init(elev=30, azim=45)
        ax.dist = 5

        # Add grid for better depth perception
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show(block=True)
        
        return pca_df
