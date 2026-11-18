# Phase 2 : le bras qui dessine

Branche : `phase-2`. À partir d'une image (PNG/JPG), le SO-ARM101 va chercher un feutre
debout dans son support, le sort, dessine les contours, remet le feutre dans le support et
rentre à la maison. Ça marche en sim (trace visible dans RViz). Pour le vrai bras
(`use_sim:=false`), voir « Passer au vrai bras ».

```
image.png ──► image_to_strokes ──► traits XY (m, repère robot)
                                        │
/draw_image (String) ──► brain_node ────┤ IK pinocchio (feutre vertical)
                                        ▼
                              /joint_command ──► driver (sim ou réel)
                              /drawing/plan, /drawing/ink, /drawing/pen ──► RViz
```

Les points 1 à 6 décrivent la première étape (feutre déjà dans la pince). Les points 7 et 8
décrivent la deuxième (support, saisie et dépose) et remplacent ce qui est dit du feutre aux points 1, 2 et 6.

## Ce qui a été fait, point par point

### 0. Branche `phase-2`
Créée depuis `main`. `main` (pick & place) n'est pas modifiée.

### 1. Le feutre dans RViz : `ros2_ws/src/so101_description/urdf/so101.urdf`
- Pas de mesh à télécharger : deux primitives URDF (un cylindre orange Ø 12 mm × 120 mm
  et une bille noire au bout) dans un link `pen_link`, fixé à `gripper_frame_link`.
- Pourquoi `gripper_frame_link` : c'est le repère que l'IK utilise déjà. Son axe z pointe
  « hors de la pince », et l'IK `down=True` l'aligne sur -Z monde, donc le feutre est vertical.
- Un link `pen_tip_link` (à 3 cm sur z) marque le bout du feutre dans les TF.
- `robot_state_publisher` publie tout ça tout seul : aucun code ajouté.

### 2. Le feutre dans MuJoCo : `sim/so101_sim/assets/so101/so101_tek5.xml`
- Un `geom` cylindre dans le body `gripper`, avec la même géométrie que dans l'URDF.
- `contype=0 conaffinity=0` : visuel uniquement, il ne touche ni la table ni la pince, donc
  la physique ne bouge pas. On le voit dans l'image de la caméra externe.
- MuJoCo ne dessine pas de trait : la trace est affichée dans RViz (point 5).

### 3. Image → traits : `ros2_ws/src/so101_brain/so101_brain/image_to_strokes.py` (nouveau)
Module pur numpy + OpenCV, sans ROS, testable seul :
1. Image en niveaux de gris, redimensionnée à 400 px sur le plus grand côté.
2. Seuil d'Otsu : les formes sombres sur fond clair deviennent un masque binaire.
3. `cv2.findContours` récupère les contours, `cv2.approxPolyDP` les simplifie, et les contours de moins de 20 px (bruit) sont jetés.
4. Passage en mètres dans le repère robot : l'image est placée dans un carré de `draw_size`
   (10 cm) centré sur `draw_center` (0.22, 0.0), en gardant ses proportions. Le haut de
   l'image va vers +X (loin du robot), la droite de l'image vers -Y.
5. Ré-échantillonnage : un point tous les 4 mm, pour que l'IK suive bien les lignes droites.
6. Ordre des traits : on prend à chaque fois le trait le plus proche (glouton), et chaque
   boucle démarre au point le plus proche du stylo, pour limiter les trajets à vide.
- Auto-check : `python3 image_to_strokes.py` génère une image avec un cercle et un carré, puis vérifie
  qu'on obtient 2 traits, que tout est dans la zone, que les boucles sont fermées, que les
  points sont espacés d'au plus 4 mm et que l'orientation est bonne.

### 4. La machine à états du dessin : `ros2_ws/src/so101_brain/so101_brain/brain_node.py`
Elle réutilise ce qui existait : `_solve_ik(down=True)`, `_send_joint_command`, `_arrived`,
`_set_state` et `_move`. Le pick & place n'a pas changé.
- Nouveau topic `/draw_image` (`std_msgs/String`, chemin de l'image dans le container).
- Avant de commencer, on vérifie par IK les 4 coins de la feuille. Si l'un d'eux est hors de
  portée, on abandonne proprement.
- États : `PEN_TRAVEL` (au-dessus du début du trait, à `pen_lift` = 2 cm), puis `PEN_DOWN`,
  puis `TRACE` (on envoie le point suivant dès que le précédent est atteint à moins de
  `trace_tolerance` = 3 mm), puis `PEN_UP`, puis le trait suivant. À la fin : `RETURN`, puis `DONE`.
- Comme le stylo est vertical, la cible IK de `gripper_frame_link` est simplement le point
  du papier décalé de `pen_length` vers le haut (`_pen_target`). Aucune frame pinocchio n'a été ajoutée.
- La pince reste fermée tout le temps. `_go_home` prend maintenant un paramètre
  `gripper_pct`, pour qu'un retour maison (fin du dessin ou cible inatteignable) ne lâche pas le feutre.
- Paramètres ROS (ce sont les réglages de calibration) : `paper_z`, `pen_length`, `pen_lift`,
  `draw_center`, `draw_size`, `trace_tolerance`.

### 5. La trace dans RViz
- `/drawing/plan` (gris) : tous les traits prévus, publiés quand l'image est reçue.
- `/drawing/ink` (noir) : la position réelle du bout du feutre, calculée par FK à partir
  de `/joint_states`, enregistrée seulement quand il est à moins de 3 mm du papier. On voit donc ce qui est
  **vraiment** dessiné, pas ce qui était commandé.
- `ros2_ws/so101.rviz` : deux displays `Marker` ont été ajoutés.
- `so101_brain/package.xml` : la dépendance `visualization_msgs` a été ajoutée.

### 6. Lancement et Docker
- `brain.launch.py` accepte maintenant `autostart`, `paper_z` et `pen_length`.
- `docker/control_entrypoint.sh` et `docker-compose.yml` lisent la variable d'env `BRAIN_AUTOSTART`
  (par défaut `true`, comme avant). Avec `false`, le brain attend `/draw_image` au lieu de
  lancer le pick & place dès que l'objet est vu. Elle est documentée dans `.env.example`.
- Images de test : `ros2_ws/images/star.png` et `ros2_ws/images/smiley.png`.
- `docker/draw.sh <image>` : lance un dessin en une commande (voir Utilisation).

### 7. Support de feutre (partie 2 : attraper le feutre)
- `hardware/pen_holder/pen_holder.py` génère `pen_holder.stl` : une base carrée de 60 × 60 × 20 mm
  (basse exprès, voir point 8),
  avec 12 languettes flexibles autour d'un trou de Ø 8 mm, et un puits traversant de Ø 12 mm (feutres type Sharpie, 10 à 11,5 mm) qui
  garde le feutre vertical. La pointe du feutre repose sur la table. Toutes les cotes sont des
  constantes en haut du script ; il vérifie la pièce lui-même avant d'écrire le STL.
  ```bash
  uv run --no-project --with manifold3d --with numpy hardware/pen_holder/pen_holder.py
  ```
- Le même STL est copié dans `so101_description/meshes/`. Dans l'URDF, c'est le link fixe
  `pen_holder_link`, placé en (0.18, 0.15, 0). Pour le déplacer, modifier le `xyz` de `pen_holder_joint`.
- MuJoCo (`scene_tek5.xml`) : body statique `pen_holder` à la même position. Le mesh sert à
  l'affichage ; les collisions passent par 4 boîtes qui laissent un puits carré de 12 mm ouvert.
  (Une collision sur le mesh utiliserait son enveloppe convexe, ce qui boucherait le trou.)
  Déplacer le support = modifier les deux fichiers.
- Sur cette branche, le cube et la boîte du pick & place ont été retirés de la scène MuJoCo, et
  `randomize_object` / `get_object_position` / `get_drop_box_position` de `SO101Sim`. Le pick & place
  reste sur `main`. Après un changement dans `sim/`, il faut reconstruire l'image : `docker-compose build control`.
- Impression : à l'envers, la plaque à languettes posée sur le plateau. Le cône à 45° sous les
  languettes évite les supports. Les languettes font 0,8 mm d'épaisseur ; en PLA, elles risquent
  de casser si on agrandit le puits pour des feutres plus gros (Posca, Velleda) ; dans ce cas, imprimer en PETG ou en TPU.

### 8. Attraper, dessiner, reposer
**Le feutre doit être court : 9 cm maximum** (en sim : Ø 11 × 88 mm). Tenu vertical dans la
pince, il ne reste qu'environ 45 mm de place au-dessus du point de saisie avant de taper dans
le moteur de la pince. Et l'IK ne garde la pince verticale (< 3°) qu'avec le point de saisie à
8 cm de haut ou moins. Un Sharpie de 136 mm ne passe donc pas.

- **L'idée :** dans le support, la pointe du feutre touche la table. Si on le saisit avec la pince
  à la hauteur `pen_length`, la pointe se retrouve à `pen_length` sous la pince. `pen_length` est
  donc la hauteur de saisie (0.05 par défaut), il n'y a rien à mesurer sur le feutre.
- **Séquence** (`brain_node.py`, mouvements verticaux au-dessus du support) :
  - `PEN_FETCH` (au-dessus du feutre, pince ouverte), `PEN_REACH` (descente à `pen_length`),
    `PEN_GRIP` (fermeture), `PEN_EXTRACT` (pointe 1 cm au-dessus du support), puis le dessin ;
  - à la fin : `PEN_STOW`, `PEN_INSERT` (pointe 2 mm au-dessus de la table), `PEN_RELEASE`
    (ouverture), `PEN_LEAVE`, puis `RETURN` et `DONE`.
  On peut relancer `/draw_image` tout de suite : le cycle se rejoue.
- **Nouveaux paramètres :** `pen_holder_xy` ([0.18, 0.15], la même position que l'URDF et la
  scène), `holder_height` (0.02) et `pen_total` (0.088, longueur du feutre). `_cb_draw_image` vérifie
  aussi par IK que le support est atteignable avant de commencer.
- **`GRASP_OFFSET` :** `gripper_frame_link` est posé sur la face de la mâchoire fixe. L'IK vise
  donc un point décalé de (-4.2, +1.3, 0) mm, l'axe du feutre serré entre les mâchoires (valeur mesurée en sim).
- **RViz :** le feutre n'est plus dans l'URDF. Le brain le publie sur `/drawing/pen` (display
  « Felt pen ») : dans la pince quand il est tenu, debout dans le support sinon.
- **MuJoCo :** le feutre est un corps libre (`pen`), tenu **seulement par le frottement**
  (pas de soudure). Pour que la pince le serre vraiment, trois réglages ont été nécessaires :
  - deux patins plats (`fixed_jaw_pad`, `moving_jaw_pad`) sur les mâchoires. Les enveloppes convexes
    des mâchoires sont à facettes et faisaient rouler le feutre hors de la pince ;
  - des bits de collision : le feutre (bit 2) ne touche que les patins, le sol et le support,
    pas les enveloppes du bras ;
  - `solref="0.004 1"` sur le feutre et les patins (contact raide : le servo serre à environ 45 N), et
    `noslip_iterations="5"` (le solveur anti-glissement de MuJoCo).
  Résultat mesuré : le feutre penche de 0,8° dans la pince et glisse de moins de 1 mm pendant un dessin.
- `autostart` / `BRAIN_AUTOSTART` ont été retirés : le brain attend toujours `/draw_image`.

## Utilisation (sim)

```bash
cd docker
docker-compose up -d control viz rviz        # RViz : http://localhost:6080/vnc.html
./draw.sh ../ros2_ws/images/smiley.png       # ou n'importe quelle image : ./draw.sh ~/Desktop/chat.png
```

**Interface web :** http://localhost:8080. On y dépose une image, ou on clique sur « Étoile » / « Smiley » ; la page n'affiche que la vue 3D de RViz, qu'on peut tourner et zoomer à la souris.
- `ros2_ws/web/server.py` (bibliothèque standard et `rclpy`, dans le container `rviz`) : il enregistre l'image sous
  `images/web_upload.png` ou `.jpg` (PNG/JPEG reconnus par leurs premiers octets, 10 Mo max) et la publie sur `/draw_image`.
  Auto-check : `docker-compose exec rviz python3 /ros2_ws/web/server.py --selftest`.
- `ros2_ws/web/index.html` affiche l'écran de RViz avec le client de noVNC (`core/rfb.js`), sans l'interface de noVNC.
- RViz tourne en plein écran (`--fullscreen`). Pour retrouver les panneaux, ouvrir noVNC (http://localhost:6080/vnc.html) et appuyer sur F11.
- Le plan gris (`/drawing/plan`) est masqué par défaut, pour que le public devine le dessin. La case
  « Afficher le plan » publie sur `/drawing/show_plan` (`std_msgs/Bool`), et le brain renvoie le plan ou un `DELETE`.
- Les PNG à fond transparent (sites d'icônes) sont posés sur du blanc avant le seuillage (`read_gray` dans `image_to_strokes.py`).
- Si le bras est occupé, la page affiche quand même « Sent… » : le brain ignore la demande (`Cannot draw now` dans les logs).
- Attention : ne pas désactiver le display « Image » dans `so101.rviz`. RViz Jazzy refuse alors toute la
  config (`Could not load display config: Invalid argument`). En plein écran, son panneau est de toute façon masqué.

`docker/draw.sh <image>` :
- copie l'image dans `ros2_ws/images/` si elle est ailleurs (le dossier est monté dans le
  container sous `/ros2_ws/images/`) ;
- vérifie que le container `control` tourne ;
- publie le chemin sur `/draw_image` ;
- affiche les logs du dessin (`Drawing ...`, `Drawing done`, erreurs). Ctrl+C arrête l'affichage des logs, pas le dessin.

Commande équivalente à la main :

```bash
docker-compose exec control bash -lc 'source /ros2_ws/install/setup.bash && \
  ros2 topic pub -w 1 --once --keep-alive 5 /draw_image std_msgs/msg/String \
  "{data: /ros2_ws/images/smiley.png}"'
```

> `--keep-alive 5` est indispensable : sans lui, `ros2 topic pub --once` se ferme avant que
> le message soit livré (transport `LARGE_DATA`), et le brain ne reçoit rien.

## Résultats vérifiés (sim, Docker/colima)
- `image_to_strokes.py` auto-check : OK.
- Smiley : 5 traits, 241 points, dessinés en 27 s. Étoile : 7 traits, 224 points, en 27 s.
- Aucun timeout, aucun point sauté, aucune cible hors de portée : chaque point a été atteint
  à moins de 3 mm (`trace_tolerance`), avec une inclinaison du feutre de 0°.
- Dans RViz, le feutre est au bout de la pince et la trace noire dessine bien la forme.

- Attraper, dessiner, reposer (point 8) : smiley et étoile, deux cycles d'affilée, sans timeout ;
  le feutre revient debout dans le support.

## Passer au vrai bras
1. Prendre un feutre de 9 cm maximum, Ø 10 à 11,5 mm, et mettre sa longueur dans `pen_total`.
2. Scotcher le support et mesurer sa position dans le repère du robot → `pen_holder_xy`.
3. Scotcher la feuille autour de (0.22, 0.0).
4. Sans feutre, envoyer `/target_position` au-dessus du support, à `z = pen_length` (0.05), pour vérifier que les
   mâchoires entourent bien la place du feutre, sans toucher le support.
5. `paper_z` : une fois le feutre saisi, envoyer `/target_position` à `z = paper_z + pen_length` et descendre par
   pas de 2 mm jusqu'à ce que le feutre touche juste le papier.
6. `USE_SIM=false` dans `docker/.env`, puis lancer avec les valeurs mesurées
   (`ros2 param set /so101_brain paper_z ...`).
7. `GRASP_OFFSET` est une valeur mesurée en sim : si les traits sont décalés d'un ou deux mm, la mesurer sur le vrai bras.
8. Si les servos n'arrivent jamais à 3 mm près (on verra des timeouts dans les logs), monter `trace_tolerance`.

## Limites connues / suite
- **Contours, pas squelette** : un trait épais dans l'image est dessiné deux fois (un passage
  par bord). Solution : squelettiser l'image (`cv2.ximgproc.thinning`) pour les dessins au trait.
- **Images en photo** : Otsu marche pour un dessin noir sur fond blanc. Pour une photo, il
  faudra Canny ou une étape de stylisation.
- **Vitesse** : on avance d'un point à la fois à 10 Hz, en attendant que chaque point soit atteint. C'est volontairement prudent ;
  pour aller plus vite, on peut monter `STATE_HZ` ou envoyer le point suivant un peu avant
  d'avoir atteint le précédent.
- **Papier plat supposé** : `paper_z` est le même partout. Si la table penche, il faudra mesurer
  3 coins et interpoler.
