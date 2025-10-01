window.plotDataPaths = {
  pattern: "{{ plot_paths.get('plot_data', '') }}",
  grouping: "{{ plot_paths.get('grouping_plot_data', '') }}",
  timeBased: "{{ plot_paths.get('time_plot_data', '') }}",
  risk: "{{ plot_paths.get('risk_plot_data', '') }}"
};

const sectionTitles = {
  "KPI Summary": "📊 KPI Summary",
  "Churn Overview": "📉 Churn Overview",
  "Demographics": "👥 Demographics",
  "Policy Info": "📄 Policy Info",
  "Risk Behavior": "⚠️ Risk Behavior",
  "Time Trends": "📆 Time Trends",
  "Segment Analysis": "📊 Segment Analysis",
  "Other": "📦 Other"
};

const sectionMap = {};

function mapSection(name) {
  return (
    name.includes("Age") || name.includes("Gender") || name.includes("Region") ? "Demographics" :
    name.includes("Policy") || name.includes("Payment") || name.includes("Vehicle") ? "Policy Info" :
    name.includes("Missed Payments") || name.includes("Support Calls") ? "Risk Behavior" :
    name.includes("Monthly") ? "Time Trends" :
    name.includes("Churn Rate Overview") || name.includes("Churn Rate by Age") ? "Churn Overview" :
    name.includes("Group") ? "Segment Analysis" :
    "Other"
  );
}

async function fetchPlotData(path) {
  if (!path) return {};
  const res = await fetch("{{ url_for('static', filename='') }}" + path);
  return await res.json();
}

(async function () {
  const allPaths = [plotDataPaths.pattern, plotDataPaths.grouping, plotDataPaths.timeBased, plotDataPaths.risk];
  let allData = {};
  for (let path of allPaths) {
    const data = await fetchPlotData(path);
    allData = { ...allData, ...data };
  }

  for (const [name, plot] of Object.entries(allData)) {
    const section = mapSection(name);
    if (!sectionMap[section]) sectionMap[section] = [];
    sectionMap[section].push({ name, plot });
  }

  const container = document.getElementById('dashboard-sections');
  for (const sectionName of Object.keys(sectionMap)) {
    const details = document.createElement('details');
    details.open = true;
    const summary = document.createElement('summary');
    summary.className = "text-lg font-semibold py-2";
    summary.textContent = sectionTitles[sectionName] || sectionName;
    details.appendChild(summary);

    const grid = document.createElement('div');
    grid.className = "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 py-2";

    for (const { name, plot } of sectionMap[sectionName]) {
      const card = document.createElement('div');
      card.className = "bg-white p-4 rounded shadow";

      card.innerHTML = `<h4 class="font-semibold mb-2">${name}</h4>`;

      if (plot.type === 'image' && plot.path) {
        const img = document.createElement('img');
        img.src = "{{ url_for('static', filename='') }}" + plot.path;
        img.alt = name;
        img.className = "w-full h-auto";
        card.appendChild(img);
      } else if (plot.type === 'bar' || plot.type === 'line') {
        const canvas = document.createElement('canvas');
        const id = 'chart-' + name.replace(/\s+/g, '-').toLowerCase();
        canvas.id = id;
        card.appendChild(canvas);
        setTimeout(() => {
          const ctx = document.getElementById(id).getContext('2d');
          new Chart(ctx, {
            type: plot.type,
            data: {
              labels: plot.labels,
              datasets: plot.datasets || [{
                label: name,
                data: plot.data,
                backgroundColor: plot.backgroundColor,
                borderColor: plot.borderColor || plot.backgroundColor,
                borderWidth: 1,
                fill: plot.fill || false
              }]
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: { legend: { display: !!plot.datasets }, tooltip: { enabled: true } }
            }
          });
        }, 0);
      }
      grid.appendChild(card);
    }

    details.appendChild(grid);
    container.appendChild(details);
  }
})();
